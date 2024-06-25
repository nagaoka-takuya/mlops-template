import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.local import LocalSession
import os
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor
import time


os.environ['AWS_DEFAULT_REGION'] = 'ap-northeast-1'
region = os.environ['AWS_DEFAULT_REGION']

# 精度監視のためCloudWatch Alarmを設定
alarm_name = 'FlowerModelQualityAlarm'
alarm_description = 'Alarm for Flower Model Quality'
alarm_action = 'arn:aws:sns:ap-northeast-1:590184069860:alarm-notification'
alarm_metric_name = 'Accuracy'
alarm_threshold = 0.8
alarm_evaluation_periods = 1
alarm_comparison_operator = 'LessThanThreshold'
alarm_period = 60
alarm_statistic = 'Minimum'
alarm_treat_missing_data = 'notBreaching'
alarm_datapoints_to_alarm = 1
cw_client = boto3.Session().client("cloudwatch")

cw_client.put_metric_alarm(
    AlarmName=alarm_name,
    AlarmDescription=alarm_description,
    ActionsEnabled=True,
    AlarmActions=[alarm_action],
    MetricName=alarm_metric_name,
    Namespace='AWS/SageMaker',
    Statistic=alarm_statistic,
    Dimensions=[
        {
            'Name': 'ModelPackageGroupName',
            'Value': 'FlowerModelPackageGroup'
        }
    ],
    Period=alarm_period,
    EvaluationPeriods=alarm_evaluation_periods,
    Threshold=alarm_threshold,
    ComparisonOperator=alarm_comparison_operator,
    TreatMissingData=alarm_treat_missing_data,
    DatapointsToAlarm=alarm_datapoints_to_alarm
)