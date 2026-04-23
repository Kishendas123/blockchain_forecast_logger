from blockchain_logger import submit_prediction_to_blockchain, get_latest_prediction

tx_hash = submit_prediction_to_blockchain(
    model_name="TestModel",
    predicted_price=70000.5,
    confidence_pct=3.2
)

print("TX Hash:", tx_hash)
print("Latest:", get_latest_prediction())