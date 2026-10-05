"""Integration boundaries: real contracts, failures, no invented perception."""
import time
import httpx
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.api import routes
from backend.app.hardware.gps import BrowserGPSProvider
from backend.app.navigation.navigator import GoogleDirectionsProvider, Navigator
from backend.app.reasoning.ocr import PPOCRv5Provider
from backend.app.reasoning.vlm import OllamaQwen3VLReasoner


def response(data, status=200):
    return httpx.Response(status, json=data, request=httpx.Request('GET', 'https://example.test'))


def test_google_route_and_geocode(monkeypatch):
    monkeypatch.setattr(httpx, 'get', lambda *a, **kw: response({'status':'OK','results':[{'geometry':{'location':{'lat':12.1,'lng':77.1}}}]}))
    def post(url, **kw):
        assert kw['json']['travelMode'] == 'WALK'
        assert kw['headers']['X-Goog-Api-Key'] == 'test-secret'
        return response({'routes':[{'distanceMeters':400,'duration':'300s','polyline':{'encodedPolyline':'abc'},'legs':[{'steps':[{'navigationInstruction':{'instructions':'Turn left'},'distanceMeters':100}]}]}]})
    monkeypatch.setattr(httpx, 'post', post)
    p=GoogleDirectionsProvider('test-secret')
    nav=Navigator(p)
    res=nav.start('Station',(12.0,77.0))
    assert res['status']=='STARTED'
    assert nav.health()['route']['distance_m']==400
    assert nav.health()['route']['duration_s']==300
    assert nav.health()['instruction']=='Turn left'
    assert 'test-secret' not in str(res)


def test_google_denied_never_guides(monkeypatch):
    p=GoogleDirectionsProvider('private-key')
    p._geocode_cache['station']=(12,77)
    monkeypatch.setattr(httpx,'post',lambda *a,**kw: response({'error':{'message':'private-key'}},403))
    nav=Navigator(p)
    res=nav.start('station',(11,76))
    assert res['status']=='UNAVAILABLE'
    assert nav.state=='UNAVAILABLE'
    assert 'private-key' not in str(res)


def test_browser_gps_expires(monkeypatch):
    gps=BrowserGPSProvider()
    assert gps.get_location()['lat'] is None
    gps.update(12,77,8)
    assert gps.health()['connected']
    gps._location['timestamp']=time.time()-31
    assert gps.get_location()['lat'] is None
    assert gps.get_location()['status']=='STALE'


def test_location_api_validation_and_configuration(monkeypatch):
    client=TestClient(app)
    gps=BrowserGPSProvider()
    monkeypatch.setattr(routes.pipeline,'gps',gps)
    assert client.post('/api/location',json={'lat':91,'lon':77,'accuracy':2}).status_code==422
    assert client.post('/api/location',json={'lat':12,'lon':77,'accuracy':2}).status_code==200
    assert client.get('/api/location').json()['lat']==12


def test_color_and_sos_use_backend(monkeypatch):
    client=TestClient(app)
    monkeypatch.setattr(routes.pipeline,'trigger_color',lambda target: {'success':False,'text':f'Cannot see {target}'})
    assert client.post('/api/v1/color',json={'target':'bottle'}).json()['text']=='Cannot see bottle'
    monkeypatch.setattr(routes.pipeline,'handle_voice_command',lambda text: {'status':'not_configured','success':False})
    assert client.post('/api/sos').json()['success'] is False
    assert 'SOS_SMTP_PASSWORD' not in str(client.get('/api/sos').json())


def test_ocr_missing_engine_never_fabricates():
    ocr=PPOCRv5Provider()
    ocr.ocr_engine=None
    result=ocr.extract_text(np.random.default_rng(1).integers(0,255,(100,100,3),dtype=np.uint8))
    assert result['has_text'] is False
    assert result['full_text']==''
    assert 'unavailable' in result['short_summary']


def test_vlm_offline_never_fabricates(monkeypatch):
    import requests
    def offline(*a,**kw):
        raise requests.exceptions.ConnectionError()
    monkeypatch.setattr(requests,'post',offline)
    result=OllamaQwen3VLReasoner().reason(np.zeros((20,20,3),dtype=np.uint8),'What is in front of me?')
    assert 'unavailable' in result
    assert 'desk' not in result


def test_map_unavailable_without_gps(monkeypatch):
    monkeypatch.setattr(routes.pipeline,'gps',BrowserGPSProvider())
    assert TestClient(app).get('/api/navigation/map').status_code==409

def test_stop_during_google_request_stays_cancelled(monkeypatch):
    p=GoogleDirectionsProvider('test')
    p._geocode_cache['station']=(12,77)
    nav=Navigator(p)
    def delayed_route(*args):
        nav.stop()
        return {'status':'OK','instruction':'Continue'}
    monkeypatch.setattr(p,'get_route',delayed_route)
    assert nav.start('station',(11,76))['status']=='CANCELLED'
    assert nav.state=='CANCELLED'
    assert nav.health()['route']=={}


def test_map_proxy_never_exposes_key(monkeypatch):
    monkeypatch.setattr(routes.settings,'GOOGLE_MAPS_API_KEY','secret-test')
    gps=BrowserGPSProvider();gps.update(12,77,3)
    monkeypatch.setattr(routes.pipeline,'gps',gps)
    def image_response(*args,**kwargs):
        assert ('key','secret-test') in kwargs['params']
        return httpx.Response(200,content=b'PNG bytes',headers={'content-type':'image/png'})
    monkeypatch.setattr(httpx,'get',image_response)
    result=TestClient(app).get('/api/navigation/map')
    assert result.status_code==200
    assert result.content==b'PNG bytes'
    assert 'secret-test' not in str(result.headers)


def test_route_geometry_offroute_distance():
    from backend.app.navigation.navigator import distance_to_route
    assert distance_to_route((0,0.005),[(0,0),(0,0.01)]) < 1
    assert distance_to_route((0.001,0.005),[(0,0),(0,0.01)]) > 100


def test_no_standby_frame_used_for_assistance(monkeypatch):
    from backend.app.services.pipeline import VisionMatePipeline
    pipe=VisionMatePipeline(camera_source='mock')
    monkeypatch.setattr(pipe.camera,'get_latest_frame',lambda:None)
    monkeypatch.setattr(pipe,'get_annotated_frame',lambda:np.zeros((20,20,3),dtype=np.uint8))
    assert pipe._latest_frame() is None
    assert pipe.trigger_ask('What is in view?')['success'] is False
    assert pipe.trigger_read()['has_text'] is False
