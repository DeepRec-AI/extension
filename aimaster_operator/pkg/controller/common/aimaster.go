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
	"encoding/json"
	"fmt"
	"strings"

	trainingv1alpha1 "github.com/DeepRec-AI/extension/aimaster_operator/api/v1alpha1"
	apicommon "github.com/DeepRec-AI/extension/aimaster_operator/pkg/api/common"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/pointer"
)

// A map from fields in yaml keys to CLI parameters.
var argNamesMap = map[string]string{
	"enableJobRestart":              "--enable-job-restart",
	"maxNumOfJobRestart":            "--max-num-of-job-restart",
	"executionMode":                 "--execution-mode",
	"maxTolerateFailureRate":        "--max-tolerate-failure-rate",
	"enableLogHangDetection":        "--enable-log-hang-detection",
	"logHangIntervalInSeconds":      "--log-hang-interval-in-seconds",
	"enablePodCompletedDetection":   "--enable-pod-completed-detection",
	"podCompletedIntervalInSeconds": "--pod-completed-interval-in-seconds",
	"enableLogDirectToNas":          "--enable-log-redirect-to-nas",
	"maxNumOfSameError":             "--max-num-of-same-error",
}

var aimasterAddedFinalizers = []string{
	"pai.training.io/preempt-protector",
}

const (
	// TODO(zhaohanyu.zhy): configure these default parameters using config instead of constants.
	defaultImage              = ""
	defaultServiceAccountName = "default"
	defaultRestartPolicy      = corev1.RestartPolicyNever
	defaultCPURequest         = "1"   // 1 cores
	defaultMemRequest         = "1Gi" // 1 GB
	defaultCPULimit           = "4"
	defaultMemLimit           = "8Gi"
	annotationAIMasterReady   = "deeprecmaster"
	envAIMasterEnv            = "DEEPRECMASTER_ADDR"
)

func GetAIMasterPodSpec(
	spec apicommon.AIMasterSpec,
	jobName, jobType, namespace string,
	community,
	kubeflow bool) *corev1.PodTemplateSpec {

	defaultSpec(&spec)
	if !*spec.Enabled {
		return nil
	}
	ret := &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "aimaster",
					Image: spec.Image,
					Args:  genAIMasterArgs(spec, community, kubeflow),
					Env: []corev1.EnvVar{
						{
							Name:  "JOB_NAME",
							Value: jobName,
						},
						{
							Name:  "JOB_TYPE",
							Value: jobType,
						},
						{
							Name:  "NAMESPACE",
							Value: namespace,
						},
					},
					ImagePullPolicy: corev1.PullAlways,
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse(defaultCPURequest),
							corev1.ResourceMemory: resource.MustParse(defaultMemRequest),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse(defaultCPULimit),
							corev1.ResourceMemory: resource.MustParse(defaultMemLimit),
						},
					},
				},
			},
			RestartPolicy:      spec.RestartPolicy,
			ServiceAccountName: spec.ServiceAccountName,
		},
	}
	return ret
}

func AddAIMasterAddrEnv(template *corev1.PodTemplateSpec, addr string) {
	for i := range template.Spec.Containers {
		exists := false
		for j := range template.Spec.Containers[i].Env {
			if template.Spec.Containers[i].Env[j].Name == envAIMasterEnv {
				exists = true
				template.Spec.Containers[i].Env[j].Value = addr
				break
			}
		}
		if !exists {
			template.Spec.Containers[i].Env = append(template.Spec.Containers[i].Env, corev1.EnvVar{
				Name:  envAIMasterEnv,
				Value: addr,
			})
		}
	}
}

func defaultSpec(spec *apicommon.AIMasterSpec) {
	if spec.Enabled == nil {
		spec.Enabled = pointer.Bool(true)
	} else if !*spec.Enabled {
		return
	}
	if spec.Image == "" {
		spec.Image = defaultImage
	}
	if spec.ServiceAccountName == "" {
		spec.ServiceAccountName = defaultServiceAccountName
	}
	if spec.RestartPolicy == "" {
		spec.RestartPolicy = defaultRestartPolicy
	}
}

func isAIMasterReady(status *apicommon.AIMasterStatus, job metav1.Object) bool {
	// AIMaster is considered ready after the pod and the service are ready
	// and AIMaster has marked itself as ready via annotation.
	return status.AIMasterPodPhase == corev1.PodRunning &&
		status.ServiceReady && job.GetAnnotations() != nil &&
		job.GetAnnotations()[annotationAIMasterReady] == "ready"
}

func genAIMasterArgs(spec apicommon.AIMasterSpec, community, kubeflow bool) (args []string) {
	args = append(args, genArgsStr(spec.JobMonitorPolicy)...)
	if spec.ExtraArgs != "" {
		tmp := strings.Split(spec.ExtraArgs, " ")
		for _, s := range tmp {
			if s != "" && s != " " {
				args = append(args, s)
			}
		}
	}
	args = append(args, "--community-k8s", fmt.Sprintf("%v", community),
		"--kubeflow", fmt.Sprintf("%v", kubeflow),
		"--aimaster-operator", "true")
	return args
}

func genArgsStr(spec interface{}) (args []string) {
	if spec == nil {
		return nil
	}
	specBytes, _ := json.Marshal(spec)
	keyValues := map[string]interface{}{}
	_ = json.Unmarshal(specBytes, &keyValues)
	for k, v := range keyValues {
		args = append(args, argNamesMap[k], fmt.Sprintf("%v", v))
	}
	return args
}

func genLabels(jobName string) map[string]string {
	return map[string]string{
		apicommon.LabelGroupName:   trainingv1alpha1.GroupVersion.Group,
		apicommon.LabelJobName:     strings.Replace(jobName, "/", "-", -1),
		apicommon.LabelReplicaType: strings.ToLower(apicommon.AIMasterRole),
	}
}
