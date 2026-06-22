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

import json
import re
import tempfile
from pathlib import Path

from verl.utils.qat.core import QATConfig, _ignore_patterns_for_vllm, is_qat_config_enabled, load_quantization_config


def test_ignore_patterns_for_vllm_preserves_re_prefix():
    patterns = [
        "lm_head",
        "re:.*in_proj_ba$",
        "re:.*in_proj_a$",
        "re:.*in_proj_b$",
    ]
    vllm_patterns = _ignore_patterns_for_vllm(patterns)
    assert vllm_patterns == patterns


def test_load_quantization_config_passes_re_prefix_to_vllm():
    layer_name = "model.layers.0.linear_attn.in_proj_ba"
    patterns = ["re:.*in_proj_ba$"]

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "nvfp4_w4a16.json"
        config_path.write_text(json.dumps({"quant_method": "compressed-tensors", "ignore": ["lm_head"]}))

        qat_config = QATConfig(
            enable=True,
            quantization_config_path=str(config_path),
            ignore_patterns=patterns,
        )
        quant_config = load_quantization_config(qat_config)

    assert quant_config["ignore"] == patterns
    assert any(re.match(pattern[3:], layer_name) for pattern in quant_config["ignore"] if pattern.startswith("re:"))


def test_is_qat_config_enabled():
    assert not is_qat_config_enabled(None)
    assert not is_qat_config_enabled({"enable": False})
    assert is_qat_config_enabled({"enable": True})
    assert is_qat_config_enabled(QATConfig(enable=True))
