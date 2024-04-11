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

package patch

import (
	"k8s.io/apimachinery/pkg/types"
)

func GetRemoveFinalizersPatch(original []string, toRemove []string) *Patch {
	if len(original) == 0 || len(toRemove) == 0 {
		return nil
	}
	toRemoveSet := map[string]struct{}{}
	for _, s := range toRemove {
		toRemoveSet[s] = struct{}{}
	}
	var newFinalizers []string
	for _, f := range original {
		if _, ok := toRemoveSet[f]; !ok {
			newFinalizers = append(newFinalizers, f)
		}
	}
	if len(newFinalizers) == len(original) {
		return nil
	}
	return &Patch{
		patchType: types.MergePatchType,
		patchData: &RemoveFinalizerPatch{Meta: &MetaDataPatch{Finalizers: newFinalizers}},
	}
}
