import boto3
import json

bd_clt = boto3.client("bedrock-runtime", region_name="us-east-1")
question = "User: Find all employees with the last name \"ERIC\" who are enrolled in a service?"

prompt = """
You are an AI SQL assistant. Generate SQL queries based on natural language prompts. Use the given schema as reference.
Generate sql statement inside a tag <SQL>. Also include finetune_llm_querygen database name with a dot in outputquery

Schema:
{
  "tables": {
    "employee": {
      "columns": {
        "employee_id": "int",
        "first_name": "varchar",
        "last_name": "varchar",
        "department_id": "int",
        "hire_date": "date"
      }
    },
    "department": {
      "columns": {
        "department_id": "int",
        "department_name": "varchar"
      }
    },
    "services": {
      "columns": {
        "service_id": "int",
        "service_name": "varchar"
      }
    },
    "enrollment": {
      "columns": {
        "employee_id": "int",
        "service_id": "int",
        "enrollment_date": "date"
      }
    }
  }
}

Examples:
User: List all employees in the IT department.
AI: SELECT * FROM "employee" e JOIN department d ON e.department_id = d.department_id WHERE d.department_name = 'IT';

User: Get all employees hired after January 1, 2021.
AI: SELECT * FROM "employee" WHERE hire_date > "2021-01-01";

User: Get all services available in the system.
AI: SELECT * FROM "services";

User: Find all employees with the last name 'Smith'.
AI: SELECT * FROM "employee" WHERE last_name = 'Smith';

User: Get the department details of employees hired before 2020.
AI: SELECT e.first_name, e.last_name, d.department_name FROM "employee" e JOIN department d ON e.department_id = d.department_id WHERE e.hire_date < '2020-01-01';

User: List all services employees have enrolled in.
AI: SELECT s.service_name, e.first_name, e.last_name FROM "employee" en JOIN services s ON en.service_id = s.service_id JOIN employee e ON en.employee_id = e.employee_id;

User: Which employees are enrolled in the Health Insurance service?
AI: SELECT e.first_name, e.last_name FROM "employee" en JOIN services s ON en.service_id = s.service_id JOIN employee e ON en.employee_id = e.employee_id WHERE s.service_name = 'Health Insurance';

User: What is the total number of employees in each department?
AI: SELECT d.department_name, COUNT(e.employee_id) AS employee_count FROM "employee" e JOIN department d ON e.department_id = d.department_id GROUP BY d.department_name;

User: Find employees who have never enrolled in a service.
AI: SELECT e.first_name, e.last_name FROM "employee" e LEFT JOIN enrollment en ON e.employee_id = en.employee_id WHERE en.employee_id IS NULL;

User: Which employees are enrolled in multiple services?
AI: SELECT e.first_name, e.last_name, COUNT(en.service_id) AS service_count FROM "enrollment" en JOIN employee e ON en.employee_id = e.employee_id GROUP BY e.first_name, e.last_name HAVING COUNT(en.service_id) > 1;

User: What is the total number of enrollments per service?
AI: SELECT s.service_name, COUNT(en.service_id) AS total_enrollments FROM "enrollment" en JOIN services s ON en.service_id = s.service_id GROUP BY s.service_name;

Now, generate the SQL query for the following request:
User: Find all employees with the last name 'Anderson'?
AI: 
"""

modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
bedrock_runtime = boto3.client('bedrock-runtime')
user_message = {"role": "user", "content": prompt}
messages = [user_message]
max_tokens = 1024
body = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": prompt,
        "messages": messages
    }
)
response = bedrock_runtime.invoke_model(body=body, modelId=modelId)
response_body = json.loads(response.get('body').read())

print(response_body)

sql_stmt = response_body['content'][0]['text']
print(sql_stmt)

import re
pattern = r'<SQL>(.*?)</SQL>'
match = re.search(pattern, sql_stmt, re.DOTALL)
print(match)

sql_content = match.group(1)
print(sql_content)

athena_client = boto3.client("athena", region_name = 'us-east-1')

bucket_name = "query-gen-hackathon"

response = athena_client.start_query_execution(QueryString= sql_content, QueryExecutionContext = {'Database': 'finetune_llm_querygen', 'Catalog': 'AwsDataCatalog'},
                                              ResultConfiguration = {'OutputLocation': "s3://query-gen-hackathon/athena-query-result", 
                                                                    })
query_id = response['QueryExecutionId']
print(query_id)

exec_status = athena_client.get_query_execution(QueryExecutionId=query_id)
print(exec_status)

execution_id = exec_status.get('QueryExecution').get('QueryExecutionId')
print(execution_id)
import time
time.sleep(3)

s3_client = boto3.client('s3')
resp = s3_client.get_object(Bucket = bucket_name, Key = f'athena-query-result/{execution_id}.csv')
print(resp['Body'])

import pandas as pd
output = pd.read_csv(resp['Body'])
print("****")
print(output)
