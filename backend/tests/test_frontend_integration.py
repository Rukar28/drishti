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

def encode_test_polyline(points):
    output=[]; previous=(0,0)
    for lat,lon in points:
        point=(round(lat*1e5),round(lon*1e5))
        for value,old in zip(point,previous):
            delta=value-old
            encoded=~(delta<<1) if delta<0 else delta<<1
            while encoded>=32:
                output.append(chr((32|(encoded&31))+63));encoded>>=5
            output.append(chr(encoded+63))
        previous=point
    return ''.join(output)


def google_fixture_provider():
    """Explicit test fixture only; never used by application factories."""
    from backend.app.navigation.navigator import MockNavigationProvider
    class Provider(MockNavigationProvider):
        name='google'
        calls=0
        def get_route(self,origin,destination):
            self.calls+=1
            return {'status':'OK','provider':'google','distance_m':222,'duration_s':160,
                    'polyline':encode_test_polyline([(12,77),(12.001,77),(12.002,77)]),
                    'instruction':'Head north', 'steps':[
                    {'instruction':'Head north','maneuver':'DEPART','distance_m':111,'end':{'latitude':12.001,'longitude':77}},
                    {'instruction':'Continue toward station','maneuver':'STRAIGHT','distance_m':111,'end':{'latitude':12.002,'longitude':77}}]}
    return Provider({'station':(12.002,77)})


def test_google_progress_steps_and_no_repeated_routes():
    provider=google_fixture_provider();nav=Navigator(provider)
    nav.start('station',(12,77))
    nav.last_update=0
    result=nav.update((12.0005,77))
    h=nav.health()
    assert 0.24<h['progress']<0.26
    assert 165<h['remaining_distance_m']<168
    assert h['remaining_duration_s']==120
    assert 54<h['distance_to_maneuver_m']<57
    assert h['next_maneuver']=='STRAIGHT'
    assert result['speak'] is False
    nav.last_update=0
    result=nav.update((12.0012,77))
    assert nav.step_index==1
    assert result['speak'] is True
    nav.last_update=0
    assert nav.update((12.0013,77))['speak'] is False
    assert provider.calls==1
    nav.last_update=0
    assert nav.update((12.002,77))['status']=='ARRIVED'
    assert nav.health()['remaining_distance_m']==0
    assert nav.health()['progress']==1


def test_navigation_pauses_and_resumes_on_gps_loss():
    nav=Navigator(google_fixture_provider());nav.start('station',(12,77))
    assert nav.update((None,None))['status']=='GPS_UNAVAILABLE'
    assert nav.state=='PAUSED_GPS'
    assert nav.health()['instruction'] is None
    assert nav.update((None,None)) is None
    result=nav.update((12.0005,77))
    assert result['status']=='GUIDING'
    assert result['speak'] is True


def test_offroute_reroutes_once_and_stop_cancels():
    provider=google_fixture_provider();nav=Navigator(provider)
    nav.start('station',(12,77));nav.last_update=0
    assert nav.update((12.0005,77.002))['status']=='REROUTED'
    assert provider.calls==2
    nav.last_update=0
    assert nav.update((12.0005,77.002))['status']=='OFF_ROUTE'
    assert provider.calls==2  # persistent off-route fixes do not flood Google
    nav.last_update=0
    assert nav.update((12.0005,77.002))['speak'] is False
    nav.last_reroute=0;nav.last_update=0
    assert nav.update((12.0005,77.002))['status']=='REROUTED'
    assert provider.calls==3
    nav.stop()
    assert nav.update((12.001,77)) is None


def test_navigation_tick_speaks_only_new_instruction(monkeypatch):
    from backend.app.services.pipeline import VisionMatePipeline
    from unittest.mock import Mock
    pipe=VisionMatePipeline(camera_source='mock')
    pipe.navigator=Navigator(google_fixture_provider())
    gps=BrowserGPSProvider();pipe.gps=gps
    gps.update(12,77,3)
    pipe.navigator.start('station',(12,77))
    pipe.tts=Mock(speech_generation=7)
    pipe.navigator.last_update=0;gps.update(12.0005,77,3)
    pipe._navigation_tick()
    pipe.tts.speak.assert_not_called()
    pipe.navigator.last_update=0;gps.update(12.0012,77,3)
    pipe._navigation_tick()
    assert pipe.tts.speak.call_count==1
    assert pipe.tts.speak.call_args.kwargs['generation_id']==7
    pipe.navigator.last_update=0;pipe._navigation_tick()
    assert pipe.tts.speak.call_count==1
    pipe.stop_navigation()
    pipe.tts.stop.assert_called_once()

def test_stale_location_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(routes.pipeline,'gps',BrowserGPSProvider())
    result=TestClient(app).post('/api/location',json={'lat':12,'lon':77,'accuracy':5,'timestamp':time.time()-60})
    assert result.status_code==422
    assert routes.pipeline.gps.get_location()['lat'] is None


def test_location_route_progress_event_and_tts_chain(monkeypatch):
    """Full local chain with explicitly controlled upstream route/GPS fixtures."""
    from unittest.mock import Mock
    nav=Navigator(google_fixture_provider());gps=BrowserGPSProvider()
    monkeypatch.setattr(routes.pipeline,'navigator',nav)
    monkeypatch.setattr(routes.pipeline,'gps',gps)
    tts=Mock(speech_generation=9)
    monkeypatch.setattr(routes.pipeline,'tts',tts)
    client=TestClient(app)
    with client.websocket_connect('/ws/events') as ws:
        assert ws.receive_json()['type']=='SYSTEM_STATUS'
        assert client.post('/api/location',json={'lat':12,'lon':77,'accuracy':3}).status_code==200
        result=client.post('/api/navigation',json={'destination':'station'}).json()
        assert result['result']['status']=='STARTED'
        nav.last_update=0
        client.post('/api/location',json={'lat':12.0012,'lon':77,'accuracy':3})
        routes.pipeline._navigation_tick()
        received=None
        for _ in range(20):
            event=ws.receive_json()
            if event['type']=='NAVIGATION_UPDATE' and event['data'].get('status')=='GUIDING':
                received=event['data'];break
        assert received is not None
        assert 0.59<received['navigation']['progress']<0.61
        assert received['navigation']['step_index']==1
        assert tts.speak.call_count==2  # start + newly reached maneuver
        stopped=client.post('/api/navigation/stop').json()
        assert stopped['status']=='stopped'
        assert nav.state=='CANCELLED'

def test_slow_ask_cannot_speak_after_stop(monkeypatch):
    from backend.app.services.pipeline import VisionMatePipeline
    from unittest.mock import Mock
    pipe=VisionMatePipeline(camera_source='mock')
    pipe.tts=Mock(speech_generation=1)
    monkeypatch.setattr(pipe.camera,'get_latest_frame',lambda:np.zeros((10,10,3),dtype=np.uint8))
    def slow_result(*args):
        pipe.tts.speech_generation=2
        return {'text':'A delayed response','success':True}
    monkeypatch.setattr(pipe.ask_handler,'ask',slow_result)
    result=pipe.trigger_ask('What is here?')
    assert result['status']=='CANCELLED'
    pipe.tts.speak.assert_not_called()


def test_webcam_does_not_return_disconnected_cached_frame():
    from backend.app.hardware.camera import DirectShowWebcam
    camera=DirectShowWebcam()
    camera._latest_frame=np.zeros((10,10,3),dtype=np.uint8)
    camera._last_frame_timestamp=time.time()
    camera._is_connected=False
    assert camera.get_latest_frame() is None
    camera._is_connected=True
    assert camera.get_latest_frame() is not None
    camera._last_frame_timestamp=time.time()-3
    assert camera.is_connected() is False
    assert camera.get_latest_frame() is None
