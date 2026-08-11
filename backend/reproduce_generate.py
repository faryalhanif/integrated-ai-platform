import traceback
from app.services.website_service import WebsiteGenerationService

service = WebsiteGenerationService()
project_id = '75709e45-47c1-4bbe-a428-95151f0e2915'
print('project exists', service.project_store.exists(project_id))
try:
    site = service.generate(project_id=project_id, prompt='Build a modern landing page for a local SaaS startup', answers={})
    print('generated', site.site_id)
    print('html length', len(site.html))
except Exception:
    traceback.print_exc()
