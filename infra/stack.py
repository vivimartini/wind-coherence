"""The live system: one private archive, one public page bucket, two scheduled tasks.

Ingest reads a new IFS or AIFS cycle and writes one Parquet object. Score runs
later, once metering has settled. A missing or failed task sends email. The
monthly budget alarm is the cost bound.
"""

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
    aws_budgets as budgets,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_events as events,
    aws_events_targets as targets,
    aws_logs as logs,
    aws_s3 as s3,
    aws_sns as sns,
    aws_sns_subscriptions as subs,
)
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[1]


class LiveStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, alert_email: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        archive = s3.Bucket(
            self,
            "Archive",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        page = s3.Bucket(
            self,
            "Page",
            block_public_access=s3.BlockPublicAccess.BLOCK_ACLS,
            public_read_access=True,
            website_index_document="index.html",
            removal_policy=RemovalPolicy.RETAIN,
        )

        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC)],
        )
        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)
        log_group = logs.LogGroup(self, "Runs", retention=logs.RetentionDays.ONE_MONTH)

        alerts = sns.Topic(self, "Alerts")
        if alert_email:
            alerts.add_subscription(subs.EmailSubscription(alert_email))

        # Fargate runs x86_64 by default; building on an arm64 Mac needs the explicit platform.
        image = ecs.ContainerImage.from_asset(str(REPO_ROOT), platform=ecr_assets.Platform.LINUX_AMD64)
        environment = {"GBWIND_ARCHIVE": archive.s3_url_for_object("live"), "GBWIND_PAGE_BUCKET": page.bucket_name}

        def task(name: str, command: list[str]) -> ecs.FargateTaskDefinition:
            definition = ecs.FargateTaskDefinition(self, name, cpu=512, memory_limit_mib=2048)
            definition.add_container(
                "app",
                image=image,
                command=command,
                environment=environment,
                logging=ecs.LogDrivers.aws_logs(stream_prefix=name, log_group=log_group),
            )
            archive.grant_read_write(definition.task_role)
            page.grant_read_write(definition.task_role)
            return definition

        ingest_aifs = task("IngestAifs", ["live", "ingest", "--source", "aifs-ens"])
        ingest_ifs = task("IngestIfs", ["live", "ingest", "--source", "ifs-ens"])
        score = task("Score", ["live", "score"])

        subnet = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)

        def schedule(name: str, definition: ecs.FargateTaskDefinition, expression: str) -> None:
            rule = events.Rule(self, name, schedule=events.Schedule.expression(expression))
            rule.add_target(
                targets.EcsTask(
                    cluster=cluster,
                    task_definition=definition,
                    subnet_selection=subnet,
                    assign_public_ip=True,
                )
            )
        # dynamical.org commits each AIFS ENS cycle about 8 h after its nominal time
        # (00 UTC by 08:20), and only the 00 UTC IFS ENS run, by about 08:40. Runs fire
        # 30 minutes later; a late cycle is caught by the next run's two-day lookback.
        schedule("AifsAfterEachCycle", ingest_aifs, "cron(50 2/6 * * ? *)")
        schedule("IfsDaily", ingest_ifs, "cron(10 9 * * ? *)")
        schedule("DailyScore", score, "cron(30 6 * * ? *)")

        failed = events.Rule(
            self,
            "TaskFailed",
            event_pattern=events.EventPattern(
                source=["aws.ecs"],
                detail_type=["ECS Task State Change"],
                detail={
                    "clusterArn": [cluster.cluster_arn],
                    "lastStatus": ["STOPPED"],
                    "containers": {"exitCode": [{"anything-but": 0}]},
                },
            ),
        )
        failed.add_target(targets.SnsTopic(alerts))

        if alert_email:
            budgets.CfnBudget(
                self,
                "MonthlyCap",
                budget=budgets.CfnBudget.BudgetDataProperty(
                    budget_type="COST",
                    time_unit="MONTHLY",
                    budget_limit=budgets.CfnBudget.SpendProperty(amount=25, unit="USD"),
                ),
                notifications_with_subscribers=[
                    budgets.CfnBudget.NotificationWithSubscribersProperty(
                        notification=budgets.CfnBudget.NotificationProperty(
                            notification_type="ACTUAL",
                            comparison_operator="GREATER_THAN",
                            threshold=80,
                        ),
                        subscribers=[
                            budgets.CfnBudget.SubscriberProperty(subscription_type="EMAIL", address=alert_email)
                        ],
                    ),
                    budgets.CfnBudget.NotificationWithSubscribersProperty(
                        notification=budgets.CfnBudget.NotificationProperty(
                            notification_type="FORECASTED",
                            comparison_operator="GREATER_THAN",
                            threshold=100,
                        ),
                        subscribers=[
                            budgets.CfnBudget.SubscriberProperty(subscription_type="EMAIL", address=alert_email)
                        ],
                    ),
                ],
            )

        CfnOutput(self, "ArchiveBucket", value=archive.bucket_name)
        CfnOutput(self, "PageBucket", value=page.bucket_name)
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "IngestAifsTaskDefinition", value=ingest_aifs.task_definition_arn)
        CfnOutput(self, "PublicSubnet", value=vpc.public_subnets[0].subnet_id)
