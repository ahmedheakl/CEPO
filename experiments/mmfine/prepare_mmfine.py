# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os

import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--local_save_dir", default="data/mmfine", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("-num_samples", "-n", type=int, default=-1, help="Number of samples to process. Default is -1, which means processing the entire dataset.")

    args = parser.parse_args()
    os.makedirs(args.local_save_dir, exist_ok=True)

    data_source = "OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking"

    dataset = datasets.load_dataset(
        data_source, split="train"
    )
    
    if args.num_samples > 0:
        dataset = dataset.select(range(args.num_samples))

    train_size = int(0.95 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))
    
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("question")
            answer = example.pop("answer")
            images = [example.pop("image")]

            data = {
                "problem": problem,
                "images": images,
                "answer": answer,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=64)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=64)

    local_save_dir = args.local_save_dir
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    


