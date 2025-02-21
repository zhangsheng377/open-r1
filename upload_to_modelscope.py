from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

YOUR_ACCESS_TOKEN = 'ea24a62f-373b-4ed1-8c17-5aad793bd6ed'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'zhangshengdong'
model_name = 'Qwen2.5-7B-Open-R1-GRPO_20250218'
model_id = f"{owner_name}/{model_name}"
api.create_model(
    model_id,
    visibility=ModelVisibility.PUBLIC,
    license=Licenses.APACHE_V2,
    chinese_name="qwen7b_OpenR1_zero模型"
)

api.upload_folder(
    repo_id=f"{owner_name}/{model_name}",
    folder_path='output/Qwen2.5-7B-Open-R1-GRPO',
    commit_message='20250218-长度与正确性奖励函数解耦',
)

# https://www.modelscope.cn/models/zhangshengdong/Qwen2.5-7B-Open-R1-GRPO
