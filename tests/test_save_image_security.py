from __future__ import annotations

import base64
from pathlib import Path

import pytest

from feishubot.ai.tools.builtins.save_image import SaveImageTool


@pytest.mark.asyncio
async def test_save_image_tool_prevents_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("feishubot.config.settings.image_local_dir", str(tmp_path))

    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
    data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"

    tool = SaveImageTool()

    # 1. 尝试使用 ../ 逃逸到 tmp_path 外部
    result = await tool.run({"data_url": data_url, "filename": "../../traversal_test.png"})
    saved_path = Path(result["path"])

    # 验证文件名被清洗为纯 basename，且仍保存在 tmp_path 内部
    assert saved_path.parent.resolve() == tmp_path.resolve()
    assert saved_path.name == "traversal_test.png"
    assert saved_path.exists()

    # 2. 尝试使用绝对路径
    result2 = await tool.run({"data_url": data_url, "filename": "C:/Windows/evil.png"})
    saved_path2 = Path(result2["path"])
    assert saved_path2.parent.resolve() == tmp_path.resolve()
    assert saved_path2.name == "evil.png"
