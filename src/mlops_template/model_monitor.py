from sagemaker.model_monitor import ModelQualityMonitor, EndpointInput

model_quality_monitor = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)

baseline_job_name = 'your-baseline-job-name'
baseline_data_uri = f's3://{bucket}/{prefix}/baseline/'

model_quality_monitor.suggest_baseline(
    job_name=baseline_job_name,
    baseline_inputs=[EndpointInput(endpoint_name=predictor.endpoint, destination='/opt/ml/processing/input')],
    problem_type='MulticlassClassification',
    ground_truth_s3_uri=baseline_data_uri,
    output_s3_uri=f's3://{bucket}/{prefix}/baseline/output'
)

monitoring_schedule_name = 'your-monitoring-schedule-name'

model_quality_monitor.create_monitoring_schedule(
    monitoring_schedule_name=monitoring_schedule_name,
    endpoint_input=EndpointInput(endpoint_name=predictor.endpoint),
    ground_truth_input=baseline_data_uri,
    problem_type='MulticlassClassification',
    schedule_cron_expression='cron(0 * ? * * *)',  # Every hour
    output_s3_uri=f's3://{bucket}/{prefix}/monitoring/output'
)

import boto3

client = boto3.client('s3')

response = client.list_objects_v2(
    Bucket=bucket,
    Prefix=f'{prefix}/monitoring/output'
)

for obj in response.get('Contents', []):
    print(obj['Key'])