import traceback
from app.core.config import get_settings
from app.services.llm_service import LLMService
from app.services.prompt_service import get_project_store

settings = get_settings()
print('DATABASE_URL=', repr(settings.DATABASE_URL))
print('STORAGE_DIR=', repr(settings.STORAGE_DIR))
print('WEBSITE_LLM_PROVIDER=', repr(settings.WEBSITE_LLM_PROVIDER))
print('WEBSITE_LLM_MODEL=', repr(settings.WEBSITE_LLM_MODEL))
print('WEBSITE_LLM_API_KEY=', repr(settings.WEBSITE_LLM_API_KEY))
print('LLM_PROVIDER=', repr(settings.LLM_PROVIDER))
print('LLM_MODEL=', repr(settings.LLM_MODEL))
print('LLM_API_KEY=', repr(settings.LLM_API_KEY))
print('WEBSITE_LLM_MAX_TOKENS=', settings.WEBSITE_LLM_MAX_TOKENS)
print('WEBSITE_LLM_TEMPERATURE=', settings.WEBSITE_LLM_TEMPERATURE)
print('ENV loaded')

try:
    model = LLMService.get_website_llm()
    print('Website LLM instance:', type(model).__name__)
except Exception:
    traceback.print_exc()

try:
    store = get_project_store()
    print('Project store type:', type(store).__name__)
    if hasattr(store, '_dir'):
        print('store dir:', store._dir)
        print('files:', [p.name for p in store._dir.glob('*.json')])
    if hasattr(store, 'exists'):
        print('project exists check for example id:', store.exists('75709e45-47c1-4bbe-a428-95151f0e2915'))
except Exception:
    traceback.print_exc()
