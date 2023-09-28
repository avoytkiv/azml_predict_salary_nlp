# azure_ml_sandbox

This projects shows how to use Azure ML to train a model and deploy it as a web service.

The data science workflow is as follows:

Install and activate the conda environment by executing the following commands:

```bash
conda env create -f environment.yml
conda activate azure_ml_sandbox
```

## Training and deploying in cloud

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

```


