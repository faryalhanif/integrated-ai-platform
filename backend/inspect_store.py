from app.services.website_service import WebsiteGenerationService
service = WebsiteGenerationService()
print('cwd=', __import__('os').getcwd())
print('project_store_dir=', service.project_store._dir)
print('project_store_files=', [p.name for p in service.project_store._dir.glob('*.json')])
for p in service.project_store._dir.glob('*.json'):
    print(p, p.exists())
print('path exists', (service.project_store._dir / '75709e45-47c1-4bbe-a428-95151f0e2915.json').exists())
print('file contents start', (service.project_store._dir / '75709e45-47c1-4bbe-a428-95151f0e2915.json').read_text()[:120])
