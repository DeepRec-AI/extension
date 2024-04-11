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

package v1alpha1

import (
	commonv1 "github.com/kubeflow/common/pkg/apis/common/v1"
	tensorflowv1 "github.com/kubeflow/training-operator/pkg/apis/tensorflow/v1"
	"github.com/DeepRec-AI/extension/aimaster_operator/pkg/api/common"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// TFJobDefaultPortName is name of the port used to communicate between PS and
	// workers.
	TFJobDefaultPortName = "tfjob-port"
	// TFJobDefaultContainerName is the name of the TFJob container.
	TFJobDefaultContainerName = "tensorflow"
	// TFJobDefaultPort is default value of the port.
	TFJobDefaultPort = 2222
	// TFJobDefaultRestartPolicy is default RestartPolicy for TFReplicaSpec.
	TFJobDefaultRestartPolicy = commonv1.RestartPolicyNever
	// TFJobKind is the kind name.
	TFJobKind = "TFJob"
	// TFJobPlural is the TensorflowPlural for TFJob.
	TFJObPlural = "tfjobs"
	// TFJobSingular is the singular for TFJob.
	TFJobSingular = "tfjob"
	// TFJobFrameworkName is the name of the ML Framework
	TFJobFrameworkName = "tensorflow"
)

// SuccessPolicy is the success policy.
type SuccessPolicy string

const (
	SuccessPolicyDefault    SuccessPolicy = ""
	SuccessPolicyAllWorkers SuccessPolicy = "AllWorkers"
)

// TFReplicaType is the type for TFReplica. Can be one of: "Chief"/"Master" (semantically equivalent),
// "Worker", "PS", or "Evaluator".

const (
	// TFJobReplicaTypePS is the type for parameter servers of distributed TensorFlow.
	TFJobReplicaTypePS commonv1.ReplicaType = "PS"

	// TFJobReplicaTypeWorker is the type for workers of distributed TensorFlow.
	// This is also used for non-distributed TensorFlow.
	TFJobReplicaTypeWorker commonv1.ReplicaType = "Worker"

	// TFJobReplicaTypeChief is the type for chief worker of distributed TensorFlow.
	// If there is "chief" replica type, it's the "chief worker".
	// Else, worker:0 is the chief worker.
	TFJobReplicaTypeChief commonv1.ReplicaType = "Chief"

	// TFJobReplicaTypeMaster is the type for master worker of distributed TensorFlow.
	// This is similar to chief, and kept just for backwards compatibility.
	TFJobReplicaTypeMaster commonv1.ReplicaType = "Master"

	// TFJobReplicaTypeEval is the type for evaluation replica in TensorFlow.
	TFJobReplicaTypeEval commonv1.ReplicaType = "Evaluator"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +resource:path=tfjob
//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
//+kubebuilder:printcolumn:name="State",type=string,JSONPath=`.status.conditions[-1:].type`
//+kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// TFJobSpec is a desired state description of the TFJob.
type TFJobSpec struct {
	tensorflowv1.TFJobSpec `json:",inline"` // Embed tensorflowv1.TFJobSpec inline

	// AIMasterSpec specifies AIMaster configurations.
	//+kubebuilder:validation:Optional
	AIMasterSpec common.AIMasterSpec `json:"aimasterSpec,omitempty"`
}

// PyTorchJob is the Schema for the pytorchjobs API
type TFJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TFJobSpec        `json:"spec,omitempty"`
	Status common.JobStatus `json:"status,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +resource:path=tfjobs
//+kubebuilder:object:root=true

// TFJobList is a list of TFJobs.
type TFJobList struct {
	// Standard type metadata.
	metav1.TypeMeta `json:",inline"`

	// Standard list metadata.
	// +optional
	metav1.ListMeta `json:"metadata,omitempty"`

	// List of TFJobs.
	Items []TFJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TFJob{}, &TFJobList{})
}
