/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package common

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/go-logr/logr"
	"github.com/DeepRec-AI/extension/aimaster_operator/pkg/api/common"
	"github.com/DeepRec-AI/extension/aimaster_operator/pkg/patch"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

type GangScheduler string

type JobControllerConfiguration struct {
	// Enable gang scheduling by volcano
	EnableGangScheduling bool
}

type JobController struct {
	client.Client
	Scheme     *runtime.Scheme
	Log        logr.Logger
	Recorder   record.EventRecorder
	Controller ControllerInterface
	Config     JobControllerConfiguration
}

// ReconcileJob reconciles job objects, AIMaster pods and services, and native job objects.
func (c *JobController) ReconcileJob(
	ctx context.Context,
	job client.Object,
	status *common.JobStatus,
	logger logr.Logger) error {

	nativeJobObj, err := c.Controller.GetNativeJob(ctx, types.NamespacedName{
		Namespace: job.GetNamespace(),
		Name:      job.GetName(),
	})
	if err != nil {
		logger.Info("Unable to fetch native job object", "reason", err)
		return err
	}
	var nativeJob metav1.Object
	if nativeJobObj != nil {
		if nativeJob, err = meta.Accessor(nativeJobObj); err != nil {
			return err
		}
	}
	// Check if the native job object (i.e., the pytorchjob.kubeflow.org) is being deleted.
	// If so, we will delete the AIMaster pod and service and our PyTorchJob object.
	if nativeJob != nil && nativeJob.GetDeletionTimestamp() != nil {
		logger.Info("The native job object is being deleted. Cleaning up the entire job")
		return c.cleanupJob(ctx, job, nativeJob, logger)
	}

	// Reconcile AIMaster pod and service and the native job object.
	if updated, err := c.Controller.PreprocessJobSpec(ctx, job); updated {
		return err
	}
	aimasterTpl := c.Controller.GetAIMasterPodSpec(job)
	oldStatus := status.DeepCopy()
	reconcileNativeJob := true
	var aimasterAddr string
	if aimasterTpl != nil {
		if aimasterAddr, err = c.reconcilePodAndService(ctx, job, aimasterTpl, status, logger); err != nil {
			return err
		}
		reconcileNativeJob = isAIMasterReady(&status.AIMasterStatus, job)
	}
	if reconcileNativeJob {
		if err := c.Controller.ReconcileNativeJob(ctx, nativeJob, job, aimasterAddr, status, logger); err != nil {
			return err
		}
	}

	// Update Status if needed.
	if !reflect.DeepEqual(*oldStatus, *status) {
		if err = c.Controller.UpdateStatus(ctx, job, status); err != nil {
			logger.Info("Unable to update status", "reason", err)
			return err
		}
	}
	return nil
}

func (c *JobController) RemovePodsFinalizers(ctx context.Context, namespacedName types.NamespacedName) error {
	pods := &corev1.PodList{}
	if err := c.List(
		ctx,
		pods,
		client.MatchingLabels(c.Controller.GenLabels(namespacedName.Name)),
		client.InNamespace(namespacedName.Namespace)); err != nil {
		return err
	}
	for _, pod := range pods.Items {
		toRemove := make([]string, len(aimasterAddedFinalizers))
		copy(toRemove, aimasterAddedFinalizers)
		toRemove = append(toRemove, common.FinalizerPreemptProtector)
		if p := patch.GetRemoveFinalizersPatch(pod.Finalizers, toRemove); p != nil {
			if err := c.Patch(ctx, &pod, p); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c *JobController) reconcilePodAndService(
	ctx context.Context,
	job client.Object,
	aimasterTpl *corev1.PodTemplateSpec,
	status *common.JobStatus,
	logger logr.Logger) (aimasterAddr string, err error) {

	// Reconcile AIMaster service.
	svc, err := c.getAIMasterService(ctx, job)
	if err != nil {
		logger.Info("Unable to fetch AIMaster service", "reason", err)
		return "", err
	}
	if svc == nil {
		logger.Info("Creating AIMaster service")
		if aimasterAddr, err = c.createAIMasterService(ctx, job, &aimasterTpl.Spec); err != nil {
			logger.Info("Unable to create AIMaster service", "reason", err)
			return "", err
		}
		logger.Info("Created AIMaster service")
	} else {
		aimasterAddr = fmt.Sprintf("%s:%d", svc.Name, svc.Spec.Ports[0].Port)
		status.AIMasterStatus.ServiceReady = true
	}

	// Reconcile AIMaster pod.
	pod, err := c.getAIMasterPod(ctx, job)
	if err != nil {
		logger.Info("Unable to fetch AIMaster pod", "reason", err)
		return "", err
	}
	if pod == nil {
		logger.Info("Creating AIMaster pod")
		if err = c.createAIMasterPod(ctx, job, aimasterTpl, aimasterAddr); err != nil {
			logger.Info("Unable to create AIMaster pod", "reason", err)
			return "", err
		}
		logger.Info("Created AIMaster pod")
	} else if err = c.reconcileExistingPod(pod, status); err != nil {
		return "", err
	}
	job.GetAnnotations()[annotationAIMasterReady] = "ready"
	return aimasterAddr, nil
}

func (c *JobController) getAIMasterPod(ctx context.Context, j metav1.Object) (*corev1.Pod, error) {
	namespacedName := types.NamespacedName{
		Namespace: j.GetNamespace(),
		Name:      strings.Replace(j.GetName(), "/", "-", -1) + "-aimaster",
	}
	pod := &corev1.Pod{}
	if err := c.Get(ctx, namespacedName, pod); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	return pod, nil
}

func (c *JobController) getAIMasterService(ctx context.Context, j metav1.Object) (*corev1.Service, error) {
	namespacedName := types.NamespacedName{
		Namespace: j.GetNamespace(),
		Name:      strings.Replace(j.GetName(), "/", "-", -1) + "-aimaster",
	}
	svc := &corev1.Service{}
	if err := c.Get(ctx, namespacedName, svc); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	return svc, nil
}

// cleanupJob deletes the job and related objects.
func (c *JobController) cleanupJob(
	ctx context.Context,
	job client.Object,
	nativeJob interface{},
	logger logr.Logger) error {

	// Delete job.
	if err := c.Controller.DeleteJob(ctx, job); err != nil {
		logger.Info("Unable to delete job", "reason", err)
		return err
	}
	logger.Info("Deleted job")
	// Remove finalizer from the native job object.
	if err := c.Controller.RemoveNativeJobFinalizer(ctx, nativeJob); err != nil {
		return err
	}
	return c.RemovePodsFinalizers(ctx, types.NamespacedName{
		Namespace: job.GetNamespace(),
		Name:      job.GetName(),
	})
}

func (c *JobController) createAIMasterPod(
	ctx context.Context,
	job client.Object,
	template *corev1.PodTemplateSpec,
	aimasterAddr string) error {

	tplCopy := template.DeepCopy()
	tplCopy.Labels = genLabels(job.GetName())
	AddAIMasterAddrEnv(tplCopy, aimasterAddr)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            strings.Replace(job.GetName(), "/", "-", -1) + "-aimaster",
			Namespace:       job.GetNamespace(),
			Labels:          tplCopy.Labels,
			Annotations:     tplCopy.Annotations,
			OwnerReferences: tplCopy.OwnerReferences,
			Finalizers:      tplCopy.Finalizers,
		},
		Spec: tplCopy.Spec,
	}
	if err := controllerutil.SetControllerReference(job, pod, c.Scheme); err != nil {
		return err
	}
	if err := c.Create(ctx, pod); err != nil {
		c.Recorder.Eventf(job, corev1.EventTypeWarning, "FailedCreatePod", "Error creating: %v", err)
		return err
	}
	c.Recorder.Eventf(job, corev1.EventTypeNormal, "SuccessfulCreatePod", "Created pod: %v", pod.Name)
	return nil
}

func (c *JobController) reconcileExistingPod(pod *corev1.Pod, jobStatus *common.JobStatus) error {
	// TODO(zhaohanyu.zhy): AIMaster pod failover.
	jobStatus.AIMasterStatus.AIMasterPodPhase = pod.Status.Phase
	return nil
}

func (c *JobController) createAIMasterService(
	ctx context.Context,
	job client.Object,
	spec *corev1.PodSpec) (host string, err error) {

	name := strings.Replace(job.GetName(), "/", "-", -1) + "-aimaster"
	labels := genLabels(job.GetName())
	port := c.getPortFromJob(spec)
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: job.GetNamespace(),
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{
				{
					Name:       c.Controller.GetDefaultContainerPortName(),
					Port:       port,
					TargetPort: intstr.FromInt(int(port)),
				},
			},
			Selector:  labels,
			ClusterIP: "None",
		},
	}
	if err = controllerutil.SetControllerReference(job, svc, c.Scheme); err != nil {
		return "", err
	}
	if err = c.Create(ctx, svc); err != nil {
		c.Recorder.Eventf(job, corev1.EventTypeWarning, "FailedCreateService", "Error creating: %v", err)
		return "", err
	}
	c.Recorder.Eventf(job, corev1.EventTypeNormal, "SuccessfulCreateService", "Created service")
	return fmt.Sprintf("%s:%d", name, port), nil
}

// getPortFromJob gets the port of job container.
func (c *JobController) getPortFromJob(spec *corev1.PodSpec) int32 {
	containers := spec.Containers
	for _, container := range containers {
		if container.Name == c.Controller.GetDefaultContainerName() {
			ports := container.Ports
			for _, port := range ports {
				if port.Name == c.Controller.GetDefaultContainerPortName() {
					return port.ContainerPort
				}
			}
		}
	}
	return c.Controller.GetDefaultPort()
}
