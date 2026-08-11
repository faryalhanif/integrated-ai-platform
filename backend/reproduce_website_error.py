import traceback
from app.services.website_service import WebsiteGenerationService
service = WebsiteGenerationService()
project_id = '75709e45-47c1-4bbe-a428-95151f0e2915'
print('exists', service.project_store.exists(project_id))
try:
    site = service.generate(project_id=project_id, prompt='Build a test site', answers={})
    print('site', site.site_id)
except Exception:
    traceback.print_exc()
