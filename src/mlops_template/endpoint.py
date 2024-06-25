import sagemaker
from sagemaker.pytorch import PyTorchModel

# SageMakerセッションの作成
sagemaker_session = sagemaker.Session()
role = 'your-sagemaker-execution-role'  # SageMaker実行ロールを指定

# モデルの作成
model = PyTorchModel(
    model_data=f's3://{bucket_name}/model/model.pth',
    role=role,
    entry_point='inference.py',
    framework_version='1.6.0',
    py_version='py3',
    source_dir=f's3://{bucket_name}/model/'
)

# デプロイ
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Cloudwatch Alarmの指定

print(f'Endpoint name: {predictor.endpoint_name}')