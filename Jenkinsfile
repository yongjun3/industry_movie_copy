pipeline {
    agent any

    environment {
        CONDA_INIT = "/var/lib/jenkins/miniconda3/etc/profile.d/conda.sh"
        CONDA_ENV_NAME = "m2"
    }

    stages {
        stage('Setup Conda Environment') {
            steps {
                sh '''#!/bin/bash
                echo "📦 Setting up Conda environment..."

                source ${CONDA_INIT}
                conda activate ${CONDA_ENV_NAME}
                pip install -r requirements.txt
                '''
            }
        }

        stage('Run Tests with Coverage') {
            steps {
                sh '''#!/bin/bash
                echo "🧪 Running tests with coverage..."
                source ${CONDA_INIT}
                conda activate ${CONDA_ENV_NAME}
                python -m pytest --cov --cov-report=term-missing
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''#!/bin/bash
                echo "🚀 Deploying Flask App (optional)..."
                source ${CONDA_INIT}
                conda activate ${CONDA_ENV_NAME}
                # python server.py
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check Console logs.'
        }
        always {
            echo '📋 Jenkins job finished.'
        }
    }
}


