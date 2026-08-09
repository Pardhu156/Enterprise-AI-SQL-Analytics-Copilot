# Future enhancement: AWS deployment

AWS deployment is **not implemented** in this project. The completed delivery boundary is GitHub Actions building and publishing the frontend and backend images to Docker Hub.

A practical future path would move the images to Amazon ECR, run Streamlit and FastAPI as separate ECS Fargate services (or use App Runner where appropriate), and migrate PostgreSQL to encrypted Amazon RDS. AWS Secrets Manager would inject the Gemini key and database credentials, while an Application Load Balancer with ACM would provide HTTPS and CloudWatch would collect logs and health/latency alarms.

For a production implementation, keep FastAPI and RDS private, use a read-only PostgreSQL application account, authenticate GitHub Actions to AWS through OIDC instead of long-lived AWS keys, and define the infrastructure with Terraform or CloudFormation. These are future improvements only; no AWS resources were provisioned or deployed.
