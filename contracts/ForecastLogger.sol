/*
This contract is a storage box on the blockchain.
Allows someone to save a prediction with details -
1. Model Name
2. Predicted Value 
3. Confidence
4. Time Stamp
5. Who Submitted It
*/


// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ForecastLogger {
    // Adding owner variable:
    address public owner;

    //constructor - runs once, when the contract is first deployed
    constructor() {
        // whoever deploys this contract becomes the owner
        owner = msg.sender;
    }

    //modifier - rule that runs before the a function
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        // _; means after this check passes, continue with the rest of the function.
        _;
    }

    // Prediction Struct stores one prediction record
    struct Prediction {
        string modelName;
        uint256 predictedValue;
        uint256 confidence;
        uint256 timestamp;
        address submittedBy;
    }

    // Creates an array of predictions called predictions & public means can be read outside of contract
    Prediction[] public predictions;

    //event - blockchain log message (log down the action done)
    event PredictionSubmitted(
        string modelName,
        uint256 predictedValue,
        uint256 confidence,
        uint256 timestamp,
        address submittedBy
    );

    //function for user to input modelname, predictedvalue, confidence
    function submitPrediction(
        //memory - temporary storage during function execution for string
        string memory _modelName,
        uint256 _predictedValue,
        uint256 _confidence
    ) public onlyOwner {
        // convert string into bytes and counts the length - must be above 0 - cannot be empty
        require(bytes(_modelName).length > 0, "Model name cannot be empty");
        // input number for confidence must be between 0 and 100 & uint256 cannot be negative!
        require(_confidence <= 100, "Confidence must be between 0 and 100");

        Prediction memory newPrediction = Prediction({
            modelName: _modelName,
            predictedValue: _predictedValue,
            confidence: _confidence,
            // Blockchain records time automatically
            timestamp: block.timestamp,
            // Gets the wallet of the address of the person calling this function (msg.sender)
            submittedBy: msg.sender
        });

        // Adding new prediction to the array (similar to appending to the list)
        predictions.push(newPrediction);

        //Broadcast that a prediction has been submitted (using emit)
        emit PredictionSubmitted(
            _modelName,
            _predictedValue,
            _confidence,
            block.timestamp,
            msg.sender
        );
    }

    //function returns how many predictions are stored
    // view - only can be viewed
    //returns (uint256) - returns only one positive number
    function getPredictionCount() public view returns (uint256) {
        return predictions.length;
    }

    function getPrediction(uint256 _index) public view
        returns (
            string memory modelName,
            uint256 predictedValue,
            uint256 confidence,
            uint256 timestamp,
            address submittedBy
        ) {
            require(_index < predictions.length, "Prediction does not exist");

            // gets prediction and stores it in a temporary variable p
            Prediction memory p = predictions[_index];

            return(
                p.modelName,
                p.predictedValue,
                p.confidence,
                p.timestamp,
                p.submittedBy
            );
        }  
        

    // read out the latest Prediction
    function getLatestPrediction() public view
    returns(
        string memory modelName,
        uint256 predictedValue,
        uint256 confidence,
        uint256 timestamp,
        address submittedBy
    ) {
        require(predictions.length > 0, "No Predictions Stored");
        // takes last update:
        Prediction memory p = predictions[predictions.length - 1];

        return(
            p.modelName,
            p.predictedValue,
            p.confidence,
            p.timestamp,
            p.submittedBy
        );
    }

}