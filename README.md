<h1 align="center">🔗 Blockchain Forecast Logger</h1>

<p align="center">
A smart contract project that stores model predictions on-chain for transparency, traceability, and auditability.
</p>

<p align="center">
Built using <b>Solidity</b>, <b>Hardhat</b>, and <b>Ethers.js</b>
</p>

---

## 📌 Overview

This project demonstrates how machine learning predictions can be recorded on the blockchain.

Each prediction includes:
- Model Name
- Predicted Value
- Confidence Score
- Timestamp (on-chain)
- Wallet Address of submitter

The goal is to create a **tamper-proof audit trail** for predictions.

---

## ⚙️ Features

- ✅ Submit predictions to the blockchain
- ✅ Retrieve total number of predictions
- ✅ Fetch latest prediction
- ✅ Input validation using `require()`
- ✅ Event logging for each submission
- ✅ Local deployment using Hardhat

---

## 🛠️ Tech Stack
- Solidity
- Hardhat (v3)
- TypeScript
- Ethers.js
- Local Hardhat Network

## 🧱 Smart Contract

Main contract: `ForecastLogger.sol`

Key functions:
```solidity
submitPrediction(...)
getPredictionCount()
getLatestPrediction()