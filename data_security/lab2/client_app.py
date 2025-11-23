"""ClientApp implementation for ArTaxOr federated learning."""

import torch
from flwr.app import ClientApp, Message, Context
from flwr.app import ArrayRecord, MetricRecord, RecordDict
from task import Net, load_data, train as train_fn, test as test_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    model = Net(num_classes=7)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data_dir = context.run_config.get("data-dir", "./artaxor_data")
    trainloader, _ = load_data(partition_id, num_partitions, data_dir)

    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net(num_classes=7)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data_dir = context.run_config.get("data-dir", "./artaxor_data")
    _, valloader = load_data(partition_id, num_partitions, data_dir)

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
