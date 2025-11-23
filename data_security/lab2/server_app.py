import torch
from flwr.app import Grid, Context, ServerApp
from flwr.app import ArrayRecord, ConfigRecord
from flwr.serverapp.strategy import FedAvg
from task import Net

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    
    global_model = Net(num_classes=7)
    arrays = ArrayRecord(global_model.state_dict())
    
    strategy = FedAvg(fraction_train=fraction_train)
    
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )
    
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "artaxor_final_model.pt")
    print(f"Model saved as 'artaxor_final_model.pt'")
