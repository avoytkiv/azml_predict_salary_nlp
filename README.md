# Predicting job salary using Neural Network model on Azure ML (NLP)

This projects shows how to use Azure ML to train a model and deploy it as a web service. 
Data is mostly unstructured, so we'll use NLP to extract features from text columns.
The goal is to predict the salary based on the job description, title etc.

**High-level steps:**

- Start with jupyter notebook as a prototype.
- Refactor notebook to scripts.
  - Clean non-essential code.
  - Use functions.
  - Add logging.
- Create all necessary Azure resources (resource group, workspace, compute cluster, etc.)
- Create and run the pipeline.
- Register the model.
- Create the endpoint.
- Test the endpoint.

**Low-level steps:**

Data science:

- [x] Split the data into train, validation, and test sets. This is the first to avoid data leakage.
- [x] Featurize categorical columns.
  - [x] One-hot encoding for categorical columns.
  - [x] Tokenize and vectorize text columns.
- [x] Transform the data into tensors.
- [x] Create a simple neural network.
  - [x] Combine vectorized (one-hot encoded) and featurized text columns (bag of words).
- [x] Train the model.
- [x] Evaluate the model.

![68747470733a2f2f6769746875622e636f6d2f79616e646578646174617363686f6f6c2f6e6c705f636f757273652f7261772f6d61737465722f7265736f75726365732f77325f636f6e765f617263682e706e67](https://github.com/avoytkiv/az_ml_predict_salary_nlp/assets/74664634/42074d3e-9f5e-4b81-8a56-4c8b8f2ff641)


Moving to Azure ML:

- [x] Create a resource group.
- [x] Create a workspace.
- [x] Create a compute cluster.
- [x] Load the data to Azure ML Datastore.
- [x] Create the components.
- [x] Create and run the pipeline.

**Setup the environment**

Install and activate the conda environment by executing the following commands:

```bash
conda env create -f environment.yml
conda activate azure_ml_sandbox
```

## Training and deploying in cloud

<img width="60%" alt="Screenshot 2023-10-02 at 15 20 37" src="https://github.com/avoytkiv/az_ml_predict_salary_nlp/assets/74664634/dc615a70-dcd0-46b2-bd9f-04146004731a">

Upload the data to Azure ML Datastore. In my case, I'm using a single file, so the `type=uri-file` is used in `data.yml`.



Create compute cluster:

```bash
az ml compute create -f cloud/cluster-cpu.yml -g <resource-groupe-name> -w <workspace-name>
```

Create the dataset we'll use to train the model:

```bash
az ml data create -f cloud/data.yml -g <resource-groupe-name> -w <workspace-name>
```

NOTE: For file more than 100MB, compress or use `azcopy`

Create the components:

```bash
az ml component create -f cloud/01_split.yml
az ml component create -f cloud/02_preprocess.yml
az ml component create -f cloud/03_train.yml
az ml component create -f cloud/04_test.yml
```

Create and run the pipeline.

```bash
run_id=$(az ml job create -f cloud/pipeline-job.yml --query name -o tsv)
```

Download the trained model

```bash
az ml job download --name $run_id --output-name "model_dir"
```

Create the Azure ML model from the output.

```bash
az ml model create --name model-pipeline-cli --version 1 --path "azureml://jobs/$run_id/outputs/model_dir" --type mlflow_model
```   

<img width="100%" alt="Screenshot 2023-10-02 at 15 41 21" src="https://github.com/avoytkiv/az_ml_predict_salary_nlp/assets/74664634/40852c97-5dad-4781-b7cd-30e844f060e7">   


Create the endpoint

```
az ml online-endpoint create -f cloud/endpoint.yml
az ml online-deployment create -f cloud/deployment.yml --all-traffic
```

Test the endpoint

```
az ml online-endpoint invoke --name endpoint-pipeline-cli --request-file test_data/images_azureml.json
```

Clean up the endpoint, to avoid getting charged.

```
az ml online-endpoint delete --name endpoint-pipeline-cli -y
```

Useful commands:

```bash
az version

# Output:
{
  "azure-cli": "2.53.0",
  "azure-cli-core": "2.53.0",
  "azure-cli-telemetry": "1.1.0",
  "extensions": {}
}
```

If ML extension is not installed:

```bash 
az extension add -n ml

# Output:

{
  "azure-cli": "2.53.0",
  "azure-cli-core": "2.53.0",
  "azure-cli-telemetry": "1.1.0",
  "extensions": 
  {
    "ml": "2.20.0"
  }
}
```

Run the help command to verify your installation and see available subcommands:

```bash
az ml -h
```

## References

- [Azure ML CLI - Manage inputs and outputs of pipeline](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-inputs-outputs-pipeline?view=azureml-api-2&tabs=cli)
- [Azure GitHub - CLI - jobs](https://github.com/Azure/azureml-examples/tree/main/cli/jobs)
- [Azure ML - Data concepts](https://learn.microsoft.com/en-us/azure/machine-learning/concept-data?view=azureml-api-2&tabs=uri-file-example%2Ccli-data-create-example#examples)





