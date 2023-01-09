## Dockerizing Models for Sagemaker

To create a custom container to deploy a model in Amazon SageMaker, you will need to create a Docker image that includes your model and any required dependencies. Here is a general outline of the steps you can follow:

1. Write a script that can load your model and serve predictions when provided with input data. This script should accept command line arguments for the model endpoint and any other required parameters.

2. Create a Dockerfile that defines the base image to use and the instructions to install any required dependencies and copy your model and serving script into the container.

3. Build the Docker image using the docker build command and the Dockerfile.

4. Test the Docker image locally to ensure that it is working correctly.

5. Push the Docker image to a container registry such as Amazon Elastic Container Registry (ECR).

6. In SageMaker, use the CreateModel API to create a model resource, specifying the URI of the Docker image in the container registry as the PrimaryContainer.

7. Use the CreateEndpointConfig API to create an endpoint configuration that specifies the model, the number of instances to use for hosting the model, and the instance type to use.

8. Use the CreateEndpoint API to create an endpoint that uses the endpoint configuration.

You can then use the endpoint to send real-time inference requests to the model.

## Registering a Model with Sagemaker
To register a model in Amazon SageMaker, we need to create a model entity in Amazon SageMaker. You can create a model using the Amazon SageMaker Python SDK, the Amazon SageMaker REST API, or the AWS Management Console.

Here is an example of how you can create a model using the Python SDK:

First, we will need to install the Amazon SageMaker Python SDK and set up your AWS credentials. You can do this using pip:

```
pip install sagemaker
```
Next, import the necessary modules and set up a SageMaker client:
```
import sagemaker
sagemaker_client = sagemaker.Session().client
```
Now we are ready to create a model. You will need to provide a name for your model, the location of the model artifacts, and the name of the Amazon Elastic Container Registry (ECR) image that contains the inference code.
Here is an example of how you can create a model using the create_model method of the SageMaker client:
```
model_name = 'my-model'
model_artifact_url = 's3://my-bucket/model.tar.gz'
image = '123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest'
response = sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': image,
        'ModelDataUrl': model_artifact_url
    }
)
```
This will create a new model with the specified name and properties. You can then use this model to deploy an endpoint and start serving predictions