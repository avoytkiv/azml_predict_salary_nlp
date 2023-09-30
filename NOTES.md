Converting from Jupyter Notebooks to python scripts was relatively seamless.

However, processing data into format consumable by the model was a bit more challenging. I had to combine categorical features encoded by one-hot encoding with processed text data. The trick was that I had to split up data into train, validation and test first to avoid data leakage. Then seperately process each of the splits. I but I managed to do it through writing classes and functions that I could reuse. For future projects, I would seperate data processing pipeline and model training pipeline into two seperate pipelines. 

Moving this to cloud was a bit harder than expected. I decided to use Azure CLI to seperate my code from azure commands. Defining yaml files turned out to be not that straightforward as with DVC pipeline. I had to do a lot of trial and error to get it right.

I still don't know how to modularize my code so that I can reuse utility functions common to different components. For example, I have a logging function used in all stages. As a temporary solution, each component comes with it's own utility functions. And some fucntions are duplicated across components. What if I want to change the logging function? I would have to change it in all components. I would like to learn how to modularize my code so that I can reuse utility functions across components. In DVC this was easy, because Git tracked all the code files while DVC tracked the data files. Azure does it's versioning differently. It tracks versions of each component, as well as data. And you can see which version was used in your pipeline runs.

Also experiments are not cached in Azure. So if you want to run the same experiment slightly differently, you have to run it from scratch - this is not computationally efficient. Especially when you want to run hundreds of experiments. In DVC, it was one command that could be run for different parameters (grid search).

Another thing worse mentioning is how to define parameters for ML pipeline. In DVC, there is a great integratin with Hydra that can compose one config file from multiple config files. This is very useful and you can be flexible with your parameters. In Azure, you have to define all parameters through arguments using `parser.add_argument`.

In general, I like how everything is organized in Azure. There are three ways how you can interact with Azure ML: Studio, CLI, SDK.


**End-to-end ML pipeline in Azure ML and GitHub Actions:**

- Convert notebook to scripts.
- Work with YAML to define a command or pipeline job.
- Run scripts as a job with the CLI v2.

- Create and assign a service principal the permissions needed to run an Azure Machine Learning job.
- Store Azure credentials securely using secrets in GitHub Secrets.
- Create a GitHub Action using YAML that uses the stored Azure credentials to run an Azure Machine Learning job.

- Run linters and unit tests with GitHub Actions.
- Integrate code checks with pull requests.

- Set up environments in GitHub.
- Use environments in GitHub Actions.
- Add approval gates to assign required reviewers before moving the model to the next environment.

- Deploy a model to a managed endpoint.
- Trigger model deployment with GitHub Actions.
- Test the deployed model.