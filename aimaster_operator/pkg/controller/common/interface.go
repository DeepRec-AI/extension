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

	"github.com/go-logr/logr"
	"github.com/DeepRec-AI/extension/aimaster_operator/pkg/api/common"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// ControllerInterface encapsulates methods for reconciling different job CRDs, which are invoked by
// the common JobController and should be implemented by the Reconcilers in controllers/xxx_controller.go.
type ControllerInterface interface {
	// GetNativeJob returns the native job object, or nil if not found.
	GetNativeJob(context.Context, types.NamespacedName) (interface{}, error)
	// GetAIMasterPodSpec tries to find AIMaster pod spec from the job's replicas spec.
	GetAIMasterPodSpec(interface{}) *corev1.PodTemplateSpec
	// PreprocessJobSpec updates the job object for enabling some feature gates.
	PreprocessJobSpec(context.Context, interface{}) (bool, error)
	// ReconcileNativeJob reconciles the native job object.
	ReconcileNativeJob(context.Context, interface{}, interface{}, string, *common.JobStatus, logr.Logger) error
	// DeleteJob deletes the job object.
	DeleteJob(context.Context, interface{}) error
	// RemoveNativeJobFinalizer removes the finalizer from the native job object.
	RemoveNativeJobFinalizer(context.Context, interface{}) error
	// UpdateStatus updates the Status of the job object.
	UpdateStatus(context.Context, interface{}, *common.JobStatus) error
	// GenLabels generates the labels that the native job operator adds to the pods.
	GenLabels(string) map[string]string
	// GetDefaultContainerName returns the default container name for the specific job CRD.
	GetDefaultContainerName() string
	// GetDefaultContainerPortName returns the default container port name for the specific job CRD.
	GetDefaultContainerPortName() string
	// GetDefaultPort returns the default port for the specific job CRD.
	GetDefaultPort() int32
}
