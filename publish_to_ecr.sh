ECR_URL=225665072512.dkr.ecr.eu-west-2.amazonaws.com
REPO_URL=${ECR_URL}/cassava-disease-detector-lambda
REMOTE_IMAGE_TAG=${REPO_URL}:v1
LOCAL_IMAGE=cassava-disease-detector-lambda

# Authenticate Docker to the Amazon ECR registry
aws ecr get-login-password \
  --region "eu-west-2" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

# Build for Lambda's platform (amd64)
docker build --platform linux/amd64 -t ${LOCAL_IMAGE} .

# Tag the local image
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}

# Push to ECR
docker push ${REMOTE_IMAGE_TAG}

echo "Image pushed to ECR repository: ${REMOTE_IMAGE_TAG}"



# ------------- Previous version of the script -------------

# ECR_URL=225665072512.dkr.ecr.eu-west-2.amazonaws.com
# REPO_URL=${ECR_URL}/cassava-disease-detector-lambda
# REMOTE_IMAGE_TAG=${REPO_URL}:v1

# # name of the local image
# LOCAL_IMAGE=cassava-disease-detector-lambda

# # Authenticate Docker to the Amazon ECR registry
# aws ecr get-login-password \
#   --region "eu-west-2" \
# | docker login \
#   --username AWS \
#   --password-stdin ${ECR_URL}



# docker build -t ${LOCAL_IMAGE} . #building the local image
# docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG} #tagging the local image with the remote image name (Meaning they are the same image but with different names)
# docker push ${REMOTE_IMAGE_TAG} #then pushing the image to the remote repo

# echo "Image pushed to ECR repository: ${REMOTE_IMAGE_TAG}"