import boto3
import uuid

def lambda_handler(event, context):
    # boto3 クライアントを初期化
    client = boto3.client('sagemaker')

    # イベントからモデル名とエンドポイント名を取得
    model_name = event['ModelName']

    endpoint_config_name="myconfig"+str(uuid.uuid1())
    initial_instance_count=1
    instance_type="ml.c4.xlarge"
    endpoint_name = event['EndpointName']  # 呼び出し元から渡される想定

   # エンドポイント設定の作成
    response = client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InstanceType': instance_type,
                'InitialInstanceCount': initial_instance_count,
                'InitialVariantWeight': 1
            }
        ]
    )
    print(response)

    # エンドポイントの更新
    response = client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    return {
        'statusCode': 200,
        'body': response
    }