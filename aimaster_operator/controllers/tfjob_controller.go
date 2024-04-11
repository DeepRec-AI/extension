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

package controllers

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/go-logr/logr"
	commonv1 "github.com/kubeflow/common/pkg/apis/common/v1"
	tensorflowv1 "github.com/kubeflow/training-operator/pkg/apis/tensorflow/v1"
	trainingv1alpha1 "github.com/DeepRec-AI/extension/aimaster_operator/api/v1alpha1"
	apicommon "github.com/DeepRec-AI/extension/aimaster_operator/pkg/api/common"
	controllercommon "github.com/DeepRec-AI/extension/aimaster_operator/pkg/controller/common"
	"github.com/DeepRec-AI/extension/aimaster_operator/pkg/patch"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

const (
	controllerName = "tfjob-controller"

	kubeflowOperatorNameLabel = commonv1.OperatorNameLabel
	kubeflowOperatorName      = "tfjob-controller"
	kubeflowJobNameLabel      = commonv1.JobNameLabel
)

func NewReconciler(mgr manager.Manager, enableGangScheduling bool) *TFJobReconciler {
	r := &TFJobReconciler{}

	r.JobController = controllercommon.JobController{
		Client:     mgr.GetClient(),
		Scheme:     mgr.GetScheme(),
		Log:        mgr.GetLogger(),
		Recorder:   mgr.GetEventRecorderFor(controllerName),
		Controller: r,
		// Config:     controllercommon.JobControllerConfiguration{EnableGangScheduling: enableGangScheduling},
	}

	return r
}

// TFJobReconciler reconciles a TFJob object
type TFJobReconciler struct {
	controllercommon.JobController
	// client.Client
	// Scheme   *runtime.Scheme
	// recorder record.EventRecorder
	// Log      logr.Logger
}

// +kubebuilder:rbac:groups=training.pai.ai,resources=tfjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=training.pai.ai,resources=tfjobs/status,verbs=get;list;update;patch
// +kubebuilder:rbac:groups=training.pai.ai,resources=tfjobs/finalizers,verbs=list;update
// +kubebuilder:rbac:groups=kubeflow.org,resources=tfjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=events,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *TFJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := r.Log.WithValues(trainingv1alpha1.Singular, req.NamespacedName)

	logger.Info("Reconciling...")

	j := &trainingv1alpha1.TFJob{}
	err := r.Get(ctx, req.NamespacedName, j)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("Job was deleted")
			// Remove the finalizer we added to the native PyTorchJob so that it can be deleted.
			if nativeJob, err := r.GetNativeJob(ctx, req.NamespacedName); err != nil {
				logger.Info("Unable to fetch native Tensorflow", "reason", err)
				return ctrl.Result{}, err
			} else if nativeJob != nil {
				if err = r.RemoveNativeJobFinalizer(ctx, nativeJob); err != nil {
					return ctrl.Result{}, err
				}
			}
			return ctrl.Result{}, r.RemovePodsFinalizers(ctx, req.NamespacedName)
		}
		logger.Info("Unable to fetch TFJob", "reason", err)
		return ctrl.Result{}, err
	}
	jobCopy := j.DeepCopy()
	if err = r.ReconcileJob(ctx, jobCopy, &jobCopy.Status, logger); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

// GetNativeJob returns the native TFJob object, or nil if not found.
func (r *TFJobReconciler) GetNativeJob(
	ctx context.Context,
	namespacedName types.NamespacedName) (interface{}, error) {

	nativeJob := &tensorflowv1.TFJob{}
	if err := r.Get(ctx, namespacedName, nativeJob); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	return nativeJob, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *TFJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&trainingv1alpha1.TFJob{}).
		Owns(&tensorflowv1.TFJob{}).
		Owns(&corev1.Pod{}).
		Owns(&corev1.Service{}).
		Complete(r)
}

func (r *TFJobReconciler) preprocessForJobMonitor(job *trainingv1alpha1.TFJob) {
	if job.Spec.AIMasterSpec.JobMonitorPolicy != nil {
		job.Annotations[apicommon.AnnotationEnableJobMonitor] = "true"
		for _, spec := range job.Spec.TFReplicaSpecs {
			found := false
			for _, f := range spec.Template.Finalizers {
				if f == apicommon.FinalizerPreemptProtector {
					found = true
					break
				}
			}
			if !found {
				spec.Template.Finalizers = append(spec.Template.Finalizers, apicommon.FinalizerPreemptProtector)
			}
		}
	}
}

func (r *TFJobReconciler) GetAIMasterPodSpec(obj interface{}) *corev1.PodTemplateSpec {
	job := obj.(*trainingv1alpha1.TFJob)
	if spec, ok := job.Spec.TFReplicaSpecs[apicommon.AIMasterRole]; ok {
		return spec.Template.DeepCopy()
	}
	return controllercommon.GetAIMasterPodSpec(
		job.Spec.AIMasterSpec, job.Name, "Tensorflow", job.Namespace, true, true)
}

func (r *TFJobReconciler) PreprocessJobSpec(ctx context.Context, obj interface{}) (updated bool, err error) {
	job := obj.(*trainingv1alpha1.TFJob)
	jobCopy := job.DeepCopy()
	if _, ok := jobCopy.Spec.TFReplicaSpecs[apicommon.AIMasterRole]; ok {
		return false, nil
	}
	// Preprocess job spec according to AIMaster spec.
	r.preprocessForJobMonitor(jobCopy)
	if !reflect.DeepEqual(job.Spec, jobCopy.Spec) || !reflect.DeepEqual(job.Annotations, jobCopy.Annotations) {
		return true, r.Update(ctx, jobCopy)
	}
	return false, nil
}

func (r *TFJobReconciler) GetAPIGroupVersionKind() schema.GroupVersionKind {
	return trainingv1alpha1.GroupVersion.WithKind(trainingv1alpha1.Kind)
}

func (r *TFJobReconciler) GetAPIGroupVersion() schema.GroupVersion {
	return trainingv1alpha1.GroupVersion
}

func (r *TFJobReconciler) GetGroupNameLabelValue() string {
	return trainingv1alpha1.GroupVersion.Group
}

// ReconcileNativeJob reconciles the native TFJob object.
func (r *TFJobReconciler) ReconcileNativeJob(
	ctx context.Context,
	nativeObj interface{},
	obj interface{},
	aimasterAddr string,
	status *apicommon.JobStatus,
	logger logr.Logger) error {

	job, ok := obj.(*trainingv1alpha1.TFJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of trainingv1alpha1.TFJob", obj)
	}
	if nativeObj == nil {
		logger.Info("Creating native TFJob")
		if err := r.createNativeTFJob(ctx, job, aimasterAddr); err != nil {
			logger.Info("Unable to create native TFJob", "reason", err)
		}
		logger.Info("Created native TFJob")
		return nil
	}
	nativeJob, ok := nativeObj.(*tensorflowv1.TFJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of tensorflowv1.TFJob", nativeObj)
	}
	if err := r.handleJobMonitor(ctx, nativeJob, job); err != nil {
		return nil
	}
	status.JobStatus = *nativeJob.Status.DeepCopy()
	return nil
}

func (r *TFJobReconciler) createNativeTFJob(
	ctx context.Context,
	j *trainingv1alpha1.TFJob,
	aimasterAddr string) error {

	jobCopy := j.DeepCopy()
	for _, spec := range jobCopy.Spec.TFReplicaSpecs {
		controllercommon.AddAIMasterAddrEnv(&spec.Template, aimasterAddr)
	}
	tfjob := &tensorflowv1.TFJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:        jobCopy.Name,
			Namespace:   jobCopy.Namespace,
			Labels:      jobCopy.Labels,
			Annotations: jobCopy.Annotations,
			Finalizers:  jobCopy.Finalizers,
		},
		Spec: jobCopy.Spec.TFJobSpec,
	}
	if _, ok := tfjob.Spec.TFReplicaSpecs[apicommon.AIMasterRole]; ok {
		delete(tfjob.Spec.TFReplicaSpecs, apicommon.AIMasterRole)
	}
	if err := controllerutil.SetControllerReference(j, tfjob, r.Scheme); err != nil {
		return err
	}
	controllerutil.AddFinalizer(tfjob, apicommon.Finalizer)
	if err := r.Create(ctx, tfjob); err != nil {
		r.Recorder.Eventf(
			j, corev1.EventTypeWarning, "FailedCreateNativeJob", "Error creating: %v", err)
		return err
	}
	r.Recorder.Eventf(j, corev1.EventTypeNormal, "SuccessfulCreateNativeJob", "Created native job")
	return nil
}

func (r *TFJobReconciler) DeleteJob(ctx context.Context, job interface{}) error {
	tfJob, ok := job.(*tensorflowv1.TFJob)
	if !ok {
		return fmt.Errorf("%v is not a type of TFJob", tfJob)
	}

	return r.Delete(ctx, tfJob)
}

// RemoveNativeJobFinalizer removes the finalizer from the native TFJob object.
func (r *TFJobReconciler) RemoveNativeJobFinalizer(ctx context.Context, obj interface{}) error {
	tfJob, ok := obj.(*tensorflowv1.TFJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of tensorflowv1.TFJob", obj)
	}
	if p := patch.GetRemoveFinalizersPatch(tfJob.Finalizers, []string{apicommon.Finalizer}); p != nil {
		if err := r.Patch(ctx, tfJob, p); err != nil {
			r.Log.Info("Unable to patch native TFJob", "reason", err)
			return err
		}
	}
	return nil
}

// UpdateStatus updates the Status of a TFJob object.
func (r *TFJobReconciler) UpdateStatus(ctx context.Context, obj interface{}, status *apicommon.JobStatus) error {
	// Update Status only when the JobStatus from the native TFJob is not empty,
	// because it contains some required fields.
	if reflect.DeepEqual((*status).JobStatus, commonv1.JobStatus{}) {
		return nil
	}
	tfJob, ok := obj.(*trainingv1alpha1.TFJob)
	if !ok {
		return fmt.Errorf("%+v is not a type of tensorflowv1.TFJob", obj)
	}
	jobCopy := tfJob.DeepCopy()
	jobCopy.Status = *status
	if err := r.Status().Update(ctx, jobCopy); err != nil {
		return err
	}
	return nil
}

func (r *TFJobReconciler) GenLabels(jobName string) map[string]string {
	return map[string]string{
		kubeflowOperatorNameLabel: kubeflowOperatorName,
		kubeflowJobNameLabel:      strings.Replace(jobName, "/", "-", -1),
	}
}

func (r *TFJobReconciler) GetDefaultContainerName() string {
	return tensorflowv1.DefaultContainerName
}

func (r *TFJobReconciler) GetDefaultContainerPortName() string {
	return tensorflowv1.DefaultPortName
}

func (r *TFJobReconciler) GetDefaultPort() int32 {
	return tensorflowv1.DefaultPort
}

func (r *TFJobReconciler) handleJobMonitor(
	ctx context.Context,
	nativeJob *tensorflowv1.TFJob,
	job *trainingv1alpha1.TFJob) error {

	if nativeJob.Annotations != nil && nativeJob.Annotations[apicommon.AnnotationRestartJob] == "true" {
		r.Log.Info("Restarting native TFJob as requested by job monitor")
		jobCopy := nativeJob.DeepCopy()
		if err := r.RemoveNativeJobFinalizer(ctx, nativeJob); err != nil {
			return err
		}
		if err := r.RemovePodsFinalizers(ctx, types.NamespacedName{
			Namespace: nativeJob.Namespace,
			Name:      nativeJob.Name,
		}); err != nil {
			return err
		}
		if err := r.Delete(ctx, nativeJob); err != nil {
			r.Log.Info("Unable to delete native TFJob", "reason", err)
			return err
		}
		newNativeJob := &tensorflowv1.TFJob{
			ObjectMeta: metav1.ObjectMeta{
				Name:        jobCopy.Name,
				Namespace:   jobCopy.Namespace,
				Labels:      jobCopy.Labels,
				Annotations: jobCopy.Annotations,
				Finalizers:  jobCopy.Finalizers,
			},
			Spec: jobCopy.Spec,
		}
		newNativeJob.Annotations[apicommon.AnnotationRestartJob] = "false"
		if err := controllerutil.SetControllerReference(job, newNativeJob, r.Scheme); err != nil {
			return err
		}
		if err := r.Create(ctx, newNativeJob); err != nil {
			r.Log.Info("Unable to create native TFJob", "reason", err)
			return err
		}
		r.Log.Info("Restarted native TFJob")
	}
	return nil
}
