import { network } from "hardhat";

async function main() {
    const { ethers } = await network.connect();

    const forecastLogger = await ethers.deployContract("ForecastLogger");
    await forecastLogger.waitForDeployment();

    console.log("ForecastLogger deployed to:", await forecastLogger.getAddress());    
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});