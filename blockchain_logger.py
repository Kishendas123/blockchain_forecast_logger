import json
import os
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3

# print("blockchain_logger loaded")


# Load .env variables
load_dotenv()

RPC_URL = os.getenv("RPC_URL")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
OWNER_ADDRESS = os.getenv("OWNER_ADDRESS")
OWNER_PRIVATE_KEY = os.getenv("OWNER_PRIVATE_KEY")

# Path to ABI file
ABI_PATH = Path("artifacts/contracts/ForecastLogger.sol/ForecastLogger.json")

# Load ABI
with open(ABI_PATH, "r") as f:
    contract_json = json.load(f)

ABI = contract_json["abi"]

# Connect to blockchain
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# Create contract instance
contract = w3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS),
    abi=ABI
)

# Submit prediction
def submit_prediction_to_blockchain(model_name: str, predicted_price: float, confidence_pct: float):

    predicted_value = int(round(predicted_price))
    confidence_value = int(round(confidence_pct))

    checksum_address = Web3.to_checksum_address(OWNER_ADDRESS)

    nonce = w3.eth.get_transaction_count(checksum_address)

    tx = contract.functions.submitPrediction(
        model_name,
        predicted_value,
        confidence_value
    ).build_transaction({
        "from": checksum_address,
        "nonce": nonce,
        "gas": 300000,
        "gasPrice": w3.to_wei("2", "gwei")
    })

    signed_tx = w3.eth.account.sign_transaction(tx, private_key=OWNER_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    return receipt.transactionHash.hex()

# Get latest prediction
def get_latest_prediction():
    result = contract.functions.getLatestPrediction().call()

    return {
        "model_name": result[0],
        "predicted_value": result[1],
        "confidence": result[2],
        "timestamp": result[3],
        "submitted_by": result[4]
    }


# Get total count
def get_prediction_count():
    return contract.functions.getPredictionCount().call()