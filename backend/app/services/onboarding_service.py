from uuid import uuid4

from app.agents.business_profile_agent import BusinessProfileAgent
from app.agents.prompt_generator import PromptGenerator
from app.schemas.business import BusinessSubmission
from app.services.llm_service import LLMService
from app.services.prompt_service import get_project_store


class OnboardingService:
    """
    Handles the complete business onboarding workflow:

    BusinessSubmission
        -> BusinessProfileAgent
            -> BusinessProfile
                -> PromptGenerator
                    -> system prompt
                        -> persisted via ProjectStore
    """

    def __init__(self):

        self.llm = LLMService.get_llm()

        self.profile_agent = BusinessProfileAgent(self.llm)

        self.prompt_generator = PromptGenerator(self.llm)

        self.project_store = get_project_store()

    def create_project(
        self,
        submission: BusinessSubmission,
    ) -> dict:

        project_id = str(uuid4())

        business_profile = self.profile_agent.run(
            submission=submission,
            project_id=project_id,
        )

        system_prompt = self.prompt_generator.run(business_profile)

        self.project_store.save_project(
            profile=business_profile,
            system_prompt=system_prompt,
        )

        return {
            "success": True,
            "project_id": project_id,
            "business_profile": business_profile,
            "system_prompt": system_prompt,
        }
