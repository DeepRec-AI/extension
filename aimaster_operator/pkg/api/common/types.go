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
	commonv1 "github.com/kubeflow/common/pkg/apis/common/v1"
	corev1 "k8s.io/api/core/v1"
)

// JobStatus represents the current observed state of the training Job.
// +k8s:deepcopy-gen=true
type JobStatus struct {
	commonv1.JobStatus `json:",inline"` // Embed commonv1.JobStatus inline
	AIMasterStatus     AIMasterStatus   `json:"aimaster,omitempty"`
}

// AIMasterStatus represents the current observed state of AIMaster.
type AIMasterStatus struct {
	// to be extended
	AIMasterPodPhase corev1.PodPhase `json:"aimasterPodPhase,omitempty"`
	ServiceReady     bool            `json:"serviceReady,omitempty"`
}

// AIMasterSpec configures AIMaster parameters.
// +k8s:deepcopy-gen=true
type AIMasterSpec struct {
	// Enabled specifies if AIMaster is enabled. Defaults to true.
	Enabled *bool `json:"enabled,omitempty"`
	// Image specifies the container image of AIMaster.
	//+kubebuilder:validation:Optional
	Image string `json:"image,omitempty"`
	// ServiceAccountName specifies the service account name of AIMaster.
	//+kubebuilder:validation:Optional
	ServiceAccountName string `json:"serviceAccountName,omitempty"`
	// RestartPolicy specifies the restart policy for AIMaster.
	// One of Always, OnFailure, Never and ExitCode.
	// Defaults to Never.
	//+kubebuilder:validation:Optional
	RestartPolicy corev1.RestartPolicy `json:"restartPolicy,omitempty"`
	// JobMonitorPolicy specifies the parameters for job monitors.
	//+kubebuilder:validation:Optional
	JobMonitorPolicy *JobMonitorPolicy `json:"jobMonitorPolicy,omitempty"`
	// ExtraArgs adds additional cmd args to the AIMaster process.
	//+kubebuilder:validation:Optional
	ExtraArgs string `json:"extraArgs,omitempty"`
}

// JobMonitorPolicy specifies the parameters for job monitors.
// +k8s:deepcopy-gen=true
type JobMonitorPolicy struct {
	//+kubebuilder:validation:Optional
	EnableJobRestart *bool `json:"enableJobRestart,omitempty"`
	//+kubebuilder:validation:Optional
	MaxNumOfJobRestart *int32 `json:"maxNumOfJobRestart,omitempty"`
	//+kubebuilder:validation:Optional
	ExecutionMode *string `json:"executionMode,omitempty"`
	//+kubebuilder:validation:Optional
	MaxToleratedFailureRate *string `json:"maxToleratedFailureRate,omitempty"`
	//+kubebuilder:validation:Optional
	EnableLogHangDetection *bool `json:"enableLogHangDetection,omitempty"`
	//+kubebuilder:validation:Optional
	LogHangIntervalInSeconds *int32 `json:"logHangIntervalInSeconds,omitempty"`
	//+kubebuilder:validation:Optional
	EnablePodCompletedDetection *bool `json:"enablePodCompletedDetection,omitempty"`
	//+kubebuilder:validation:Optional
	PodCompletedIntervalInSeconds *int32 `json:"podCompletedIntervalInSeconds,omitempty"`
	//+kubebuilder:validation:Optional
	EnableLogDirectToNas *bool `json:"enableLogDirectToNas,omitempty"`
	//+kubebuilder:validation:Optional
	MaxNumOfSameError *int32 `json:"maxNumOfSameError,omitempty"`
}
