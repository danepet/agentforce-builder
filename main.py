import os
from typing import Dict, List, Optional, Any, Union
import json
import tempfile
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import AgentForce SDK
from agent_sdk import Agentforce
from agent_sdk.core.auth import BasicAuth
from agent_sdk.utils.agent_utils import AgentUtils
from agent_sdk.core.prompt_template_utils import PromptTemplateUtils
from agent_sdk.models.agent import Agent
from agent_sdk.models.topic import Topic
from agent_sdk.models.action import Action, Input, Output
from agent_sdk.models.system_message import SystemMessage
from agent_sdk.models.variable import Variable

# Create FastAPI app
app = FastAPI(title="AgentForce Backend API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request/response
class AuthCredentials(BaseModel):
    username: str
    password: str
    security_token: Optional[str] = None
    domain: str = "login"

class AuthResponse(BaseModel):
    session_id: str
    instance_url: str

class AgentResponse(BaseModel):
    agent: Dict[str, Any]

class AgentsResponse(BaseModel):
    agents: List[Dict[str, Any]]

class SuccessResponse(BaseModel):
    success: bool

class ExportResponse(BaseModel):
    exportData: Dict[str, Any]

class TemplateResponse(BaseModel):
    template: Dict[str, Any]

class TemplatesResponse(BaseModel):
    templates: List[Dict[str, Any]]

class DeploymentResponse(BaseModel):
    deploymentResult: Dict[str, Any]

class ApexClassResponse(BaseModel):
    apexClass: Dict[str, Any]

# Helper functions
def initialize_agentforce(auth: AuthCredentials) -> Agentforce:
    """Initialize Agentforce client with the given credentials."""
    basic_auth = BasicAuth(
        username=auth.username,
        password=auth.password,
        security_token=auth.security_token,
        domain=auth.domain
    )
    
    return Agentforce(auth=basic_auth)

# Routes
@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "AgentForce Backend API is running"}

@app.post("/auth", response_model=AuthResponse)
async def authenticate(credentials: AuthCredentials):
    """Authenticate with Salesforce and return session details."""
    try:
        agentforce = initialize_agentforce(credentials)
        return {
            "session_id": agentforce.session_id,
            "instance_url": agentforce.instance_url
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agents/list", response_model=AgentsResponse)
async def list_agents(auth_credentials: AuthCredentials):
    """List all agents in the Salesforce org."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        # Note: The SDK doesn't have a direct listAgents method,
        # so we would need to implement this using the Salesforce metadata API
        # This is a simplified example
        agents = []  # This would be fetched from Salesforce
        return {"agents": agents}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, auth_credentials: AuthCredentials):
    """Get a specific agent by ID."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not agent_id:
            raise HTTPException(status_code=400, detail="Agent ID is required")
        
        agent = agentforce.retrieve(agent_id)
        return {"agent": agent.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agents/create", response_model=AgentResponse)
async def create_agent(agent_data: Dict[str, Any] = Body(...), auth_credentials: AuthCredentials = Body(...)):
    """Create a new agent in Salesforce."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not agent_data:
            raise HTTPException(status_code=400, detail="Agent data is required")
        
        # Convert dict to Agent object
        agent = Agent(
            name=agent_data.get('name'),
            description=agent_data.get('description'),
            agent_type=agent_data.get('agent_type', 'External'),
            company_name=agent_data.get('company_name')
        )
        
        # Add sample utterances
        if 'sample_utterances' in agent_data:
            agent.sample_utterances = agent_data.get('sample_utterances')
        
        # Add system messages
        if 'system_messages' in agent_data:
            agent.system_messages = [
                SystemMessage(
                    message=msg.get('message'),
                    msg_type=msg.get('msg_type', 'system')
                )
                for msg in agent_data.get('system_messages')
            ]
        
        # Add variables
        if 'variables' in agent_data:
            agent.variables = [
                Variable(
                    name=var.get('name'),
                    data_type=var.get('data_type'),
                    default_value=var.get('default_value', None),
                    var_type=var.get('var_type', 'custom')
                )
                for var in agent_data.get('variables')
            ]
        
        # Add topics
        if 'topics' in agent_data:
            topics = []
            for topic_data in agent_data.get('topics'):
                topic = Topic(
                    name=topic_data.get('name'),
                    description=topic_data.get('description'),
                    scope=topic_data.get('scope')
                )
                
                # Add instructions
                if 'instructions' in topic_data:
                    topic.instructions = topic_data.get('instructions')
                
                # Add actions
                if 'actions' in topic_data:
                    actions = []
                    for action_data in topic_data.get('actions'):
                        action = Action(
                            name=action_data.get('name'),
                            description=action_data.get('description')
                        )
                        
                        # Add inputs
                        if 'inputs' in action_data:
                            action.inputs = [
                                Input(
                                    name=input_data.get('name'),
                                    description=input_data.get('description'),
                                    data_type=input_data.get('data_type'),
                                    required=input_data.get('required', True)
                                )
                                for input_data in action_data.get('inputs')
                            ]
                        
                        # Add example output
                        if 'example_output' in action_data:
                            action.example_output = action_data.get('example_output')
                        
                        actions.append(action)
                    
                    topic.actions = actions
                
                topics.append(topic)
            
            agent.topics = topics
        
        # Create the agent in Salesforce
        result = agentforce.create(agent)
        
        # Return the created agent
        return {"agent": {**agent.to_dict(), "id": result.get('id')}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agents/{agent_id}/update", response_model=AgentResponse)
async def update_agent(agent_id: str, agent_data: Dict[str, Any] = Body(...), auth_credentials: AuthCredentials = Body(...)):
    """Update an existing agent in Salesforce."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not agent_id or not agent_data:
            raise HTTPException(status_code=400, detail="Agent ID and data are required")
        
        # Retrieve the existing agent
        agent = agentforce.retrieve(agent_id)
        
        # Update agent properties
        for key, value in agent_data.items():
            if key not in ['topics', 'system_messages', 'variables', 'sample_utterances']:
                setattr(agent, key, value)
        
        # Update sample utterances
        if 'sample_utterances' in agent_data:
            agent.sample_utterances = agent_data.get('sample_utterances')
        
        # Update system messages
        if 'system_messages' in agent_data:
            agent.system_messages = [
                SystemMessage(
                    message=msg.get('message'),
                    msg_type=msg.get('msg_type', 'system')
                )
                for msg in agent_data.get('system_messages')
            ]
        
        # Update variables
        if 'variables' in agent_data:
            agent.variables = [
                Variable(
                    name=var.get('name'),
                    data_type=var.get('data_type'),
                    default_value=var.get('default_value', None),
                    var_type=var.get('var_type', 'custom')
                )
                for var in agent_data.get('variables')
            ]
        
        # Update topics
        if 'topics' in agent_data:
            topics = []
            for topic_data in agent_data.get('topics'):
                topic = Topic(
                    name=topic_data.get('name'),
                    description=topic_data.get('description'),
                    scope=topic_data.get('scope')
                )
                
                # Add instructions
                if 'instructions' in topic_data:
                    topic.instructions = topic_data.get('instructions')
                
                # Add actions
                if 'actions' in topic_data:
                    actions = []
                    for action_data in topic_data.get('actions'):
                        action = Action(
                            name=action_data.get('name'),
                            description=action_data.get('description')
                        )
                        
                        # Add inputs
                        if 'inputs' in action_data:
                            action.inputs = [
                                Input(
                                    name=input_data.get('name'),
                                    description=input_data.get('description'),
                                    data_type=input_data.get('data_type'),
                                    required=input_data.get('required', True)
                                )
                                for input_data in action_data.get('inputs')
                            ]
                        
                        # Add example output
                        if 'example_output' in action_data:
                            action.example_output = action_data.get('example_output')
                        
                        actions.append(action)
                    
                    topic.actions = actions
                
                topics.append(topic)
            
            agent.topics = topics
        
        # Update the agent in Salesforce
        result = agentforce.update(agent)
        
        # Return the updated agent
        return {"agent": agent.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agents/{agent_id}/delete", response_model=SuccessResponse)
async def delete_agent(agent_id: str, auth_credentials: AuthCredentials):
    """Delete an agent from Salesforce."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not agent_id:
            raise HTTPException(status_code=400, detail="Agent ID is required")
        
        result = agentforce.delete(agent_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agents/{agent_id}/export", response_model=ExportResponse)
async def export_agent(agent_id: str, auth_credentials: AuthCredentials, format_type: str = "json"):
    """Export an agent from Salesforce."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not agent_id:
            raise HTTPException(status_code=400, detail="Agent ID is required")
        
        # Create a temporary directory to store the exported agent
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, f"agent_{agent_id}")
            
            # Export the agent
            agentforce.export(agent_id, export_path, format=format_type)
            
            # Read the exported data
            export_data = {}
            
            if format_type == 'json':
                # Read the JSON file
                with open(f"{export_path}.json", "r") as f:
                    export_data = json.load(f)
            else:
                # Read the directory structure
                for root, dirs, files in os.walk(export_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, export_path)
                        
                        with open(file_path, "r") as f:
                            content = f.read()
                            
                            try:
                                # Try to parse as JSON
                                export_data[relative_path] = json.loads(content)
                            except:
                                # Store as text
                                export_data[relative_path] = content
            
            return {"exportData": export_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agents/import", response_model=AgentResponse)
async def import_agent(
    agent_data: Dict[str, Any] = Body(...), 
    auth_credentials: AuthCredentials = Body(...),
    format_type: str = Body(default="json")
):
    """Import an agent to Salesforce."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not agent_data:
            raise HTTPException(status_code=400, detail="Agent data is required")
        
        # Create a temporary directory to store the agent data
        with tempfile.TemporaryDirectory() as temp_dir:
            if format_type == 'json':
                # Write the JSON file
                agent_path = os.path.join(temp_dir, "agent.json")
                with open(agent_path, "w") as f:
                    json.dump(agent_data, f)
            else:
                # Create the directory structure
                for path, content in agent_data.items():
                    file_path = os.path.join(temp_dir, path)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    with open(file_path, "w") as f:
                        if isinstance(content, (dict, list)):
                            json.dump(content, f)
                        else:
                            f.write(content)
                
                agent_path = temp_dir
            
            # Import the agent
            agent = agentforce.import_agent(agent_path)
            result = agentforce.create(agent)
            
            # Return the imported agent
            return {"agent": {**agent.to_dict(), "id": result.get('id')}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/templates/list", response_model=TemplatesResponse)
async def list_templates(auth_credentials: AuthCredentials):
    """List all prompt templates in the Salesforce org."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        base = agentforce.base
        prompt_utils = PromptTemplateUtils(base.sf)
        
        # Note: The SDK doesn't have a direct listTemplates method,
        # so we would need to implement this using the Salesforce metadata API
        # This is a simplified example
        templates = []  # This would be fetched from Salesforce
        return {"templates": templates}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/templates/create", response_model=TemplateResponse)
async def create_template(template_data: Dict[str, Any] = Body(...), auth_credentials: AuthCredentials = Body(...)):
    """Create a new prompt template in Salesforce."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not template_data:
            raise HTTPException(status_code=400, detail="Template data is required")
        
        base = agentforce.base
        prompt_utils = PromptTemplateUtils(base.sf)
        
        # Create a temporary directory to store the template
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the template file
            template_path = os.path.join(temp_dir, "template.json")
            with open(template_path, "w") as f:
                json.dump(template_data, f)
            
            # Deploy the template
            result = prompt_utils.deploy_prompt_template(template_path)
            
            return {"template": {**template_data, "deploymentResult": result}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/templates/generate", response_model=TemplateResponse)
async def generate_template(
    name: str = Body(...),
    description: str = Body(...),
    auth_credentials: AuthCredentials = Body(...),
    object_names: List[str] = Body(default=[]),
    model: str = Body(default="gpt-4")
):
    """Generate a new prompt template."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not name or not description:
            raise HTTPException(status_code=400, detail="Name and description are required")
        
        base = agentforce.base
        prompt_utils = PromptTemplateUtils(base.sf)
        
        # Create a temporary directory for the output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate the template
            template = prompt_utils.generate_prompt_template(
                name=name,
                description=description,
                object_names=object_names,
                output_dir=temp_dir,
                model=model
            )
            
            # Save the template
            template_path = prompt_utils.save_prompt_template(template, temp_dir)
            
            # Read the template file
            with open(template_path, "r") as f:
                template_data = json.load(f)
            
            return {"template": template_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/templates/{template_id}/deploy", response_model=DeploymentResponse)
async def deploy_template(
    template_id: str,
    auth_credentials: AuthCredentials = Body(...),
    validate_only: bool = Body(default=False)
):
    """Deploy a prompt template to Salesforce."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not template_id:
            raise HTTPException(status_code=400, detail="Template ID is required")
        
        base = agentforce.base
        prompt_utils = PromptTemplateUtils(base.sf)
        
        # Deploy the template
        result = prompt_utils.deploy_prompt_template(template_id, validate_only=validate_only)
        
        return {"deploymentResult": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/templates/{template_id}/tune", response_model=TemplateResponse)
async def tune_template(
    template_id: str,
    description: str = Body(...),
    auth_credentials: AuthCredentials = Body(...),
    model: str = Body(default="gpt-4")
):
    """Tune a prompt template for a specific model."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not template_id or not description:
            raise HTTPException(status_code=400, detail="Template ID and description are required")
        
        base = agentforce.base
        prompt_utils = PromptTemplateUtils(base.sf)
        
        # Create a temporary directory for the output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Tune the template
            tuned_template = prompt_utils.tune_prompt_template(
                template_path=template_id,
                description=description,
                model=model,
                output_dir=temp_dir
            )
            
            # Save the tuned template
            template_path = prompt_utils.save_prompt_template(tuned_template, temp_dir)
            
            # Read the template file
            with open(template_path, "r") as f:
                template_data = json.load(f)
            
            return {"template": template_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/apex/create", response_model=ApexClassResponse)
async def create_apex_class(
    topic: Dict[str, Any] = Body(...),
    action: Dict[str, Any] = Body(...),
    auth_credentials: AuthCredentials = Body(...),
    output_dir: Optional[str] = Body(default=None)
):
    """Create an Apex class from a topic and action."""
    try:
        agentforce = initialize_agentforce(auth_credentials)
        
        if not topic or not action:
            raise HTTPException(status_code=400, detail="Topic and action are required")
        
        # Convert dict to Topic object
        topic_obj = Topic.model_validate(topic)
        
        # Convert dict to Action object
        action_obj = Action.model_validate(action)
        
        # Create a temporary directory if not provided
        temp_dir = None
        if not output_dir:
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = temp_dir.name
        
        # Create the Apex class
        class_path = agentforce.create_apex_class(topic_obj, action_obj, output_dir)
        
        # Read the Apex class file
        with open(class_path, "r") as f:
            apex_code = f.read()
        
        # Clean up temporary directory
        if temp_dir:
            temp_dir.cleanup()
        
        return {"apexClass": {
            "name": os.path.basename(class_path),
            "code": apex_code,
            "path": class_path
        }}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)