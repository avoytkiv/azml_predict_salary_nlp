# Predicting salary (NLP)

This projects shows how to use Azure ML to train a model and deploy it as a web service.

The data science workflow is as follows:

Install and activate the conda environment by executing the following commands:

```bash
conda env create -f environment.yml
conda activate azure_ml_sandbox
```

## Training and deploying in cloud

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





