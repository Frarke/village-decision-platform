from copy import deepcopy
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import shape

from api import app
from model.input_package import build_input_package, read_input_package
from model.service import load_geographic_sample, optimize_request, prepare_request
from model.geographic_map import result_map_html


@pytest.fixture(scope='module')
def request_data():
    return load_geographic_sample()


@pytest.fixture(scope='module')
def result(request_data):
    return optimize_request(request_data)


def test_separate_geotiffs_csv_and_spatial_bindings_roundtrip(request_data):
    content=build_input_package(request_data)
    with ZipFile(BytesIO(content)) as z:
        assert {'risk.tif','vulnerability.tif','engineering_profiles.csv','decision_context.geojson'} <= set(z.namelist())
        assert not any('governance_unit' in n for n in z.namelist())
    loaded=read_input_package(content)
    actual=prepare_request(loaded)
    expected=prepare_request(request_data)
    assert actual['governance_units']==expected['governance_units']
    assert actual['missing_risk_weight_ids']==[]
    assert len(actual['spatial_binding_trace'])==4


def test_units_are_irregular_and_change_when_upstream_cells_change(request_data):
    original=prepare_request(request_data)
    assert original['spatial_metrics']['priority_features']==273
    assert original['spatial_metrics']['clipped_features']==1911
    assert all(len(shape(f['geometry']).exterior.coords)>15 for f in original['governance_units']['features'])
    request=deepcopy(request_data)
    request.pop('spatial_bindings')
    for r,row in enumerate(request['spatial_layers']['layers']['profile_id']):
        for c,pid in enumerate(row):
            if pid==1:
                request['spatial_layers']['layers']['risk'][r][c]=1
                request['spatial_layers']['layers']['distance_m'][r][c]=500
    changed=prepare_request(request)
    assert changed['spatial_metrics']['governance_unit_count']==3
    assert changed['spatial_metrics']['priority_area_m2']<original['spatial_metrics']['priority_area_m2']


@pytest.mark.parametrize('mutation,match',[
    (lambda r:r['spatial_bindings']['features'][0]['geometry'].update(coordinates=[0,0]),'对应0'),
    (lambda r:r['spatial_bindings']['features'][1]['geometry'].update(coordinates=r['spatial_bindings']['features'][0]['geometry']['coordinates']),'多个'),
    (lambda r:r['spatial_bindings']['features'].pop(0),'缺少显式'),
])
def test_spatial_binding_does_not_guess_missing_or_ambiguous_inputs(request_data,mutation,match):
    request=deepcopy(request_data);mutation(request)
    with pytest.raises(ValueError,match=match):prepare_request(request)


def test_package_grid_mismatch_is_reported(request_data):
    content=build_input_package(request_data)
    with ZipFile(BytesIO(content)) as z:
        files={n:z.read(n) for n in z.namelist()}
    from rasterio.io import MemoryFile
    from affine import Affine
    with MemoryFile() as mem:
        with mem.open(driver='GTiff',height=1,width=1,count=1,dtype='float64',crs='EPSG:32650',transform=Affine(1,0,1,0,-1,1)) as ds:
            ds.write(np.array([[.5]]),1)
        files['vulnerability.tif']=mem.read()
    buffer=BytesIO()
    with ZipFile(buffer,'w',ZIP_DEFLATED) as z:
        for n,v in files.items():z.writestr(n,v)
    with pytest.raises(ValueError,match='vulnerability.tif.*不同'):read_input_package(buffer.getvalue())


def test_placements_match_each_solution_and_stay_inside_units(result):
    configurations=[]
    for solution in result['representative_solutions']:
        fc=result['map_result']['measure_placements'][solution['label']]
        units={f['properties']['unit_id']:f for f in result['map_result']['solutions'][solution['label']]['features']}
        expected={(d['unit_id'],mid) for d in solution['unit_details'] for mid in d['measure_ids'].split('+') if mid!='NONE'}
        assert {(f['properties']['unit_id'],f['properties']['measure_id']) for f in fc['features']}==expected
        configurations.append(expected)
        for f in fc['features']:
            assert shape(units[f['properties']['unit_id']]['geometry']).covers(shape(f['geometry']))
            assert f['properties']['is_design_location'] is False
    assert any(c!=configurations[0] for c in configurations[1:])
    html=result_map_html(result)
    assert 'World_Imagery' in html and 'OpenStreetMap' in html
    assert '已有评价' in html and 'measure_placements' in html and '单元级配置示意' in html


def test_uploaded_text_cannot_close_map_script(result):
    changed=deepcopy(result)
    changed['provenance']['village']['name']='</script><script>alert(1)</script>'
    assert '</script><script>alert(1)' not in result_map_html(changed)


def test_api_one_step_package_returns_same_solution_and_embeddable_html(request_data,result):
    client=TestClient(app)
    data=build_input_package(request_data)
    response=client.post('/optimize-package',content=data,headers={'Content-Type':'application/zip'})
    assert response.status_code==200,response.text
    assert response.json()['pareto_front']==result['pareto_front']
    html=client.post('/optimize-package?view=html',content=data)
    assert html.status_code==200 and html.headers['content-type'].startswith('text/html')
    assert 'measure-marker' in html.text
    assert client.get('/integration').status_code==200
    assert client.get('/symbols/M05.svg').headers['content-type'].startswith('image/svg+xml')
    assert client.post('/optimize-package',content=b'not a zip').status_code==422
    assert client.post('/optimize-package?total_budget=-1',content=data).status_code==422
    limited=client.post('/optimize-package?total_budget=400',content=data)
    assert limited.status_code==200,limited.text
    limited_result=limited.json()
    assert limited_result['provenance']['solver_config']['total_budget']==400
    assert limited_result['input_hash']!=response.json()['input_hash']
    assert all(s['total_cost']<=400 for s in limited_result['representative_solutions'])


def test_package_duplicate_names_rejected():
    buffer=BytesIO()
    with pytest.warns(UserWarning,match='Duplicate name'):
        with ZipFile(buffer,'w') as z:
            z.writestr('project.json','{}');z.writestr('project.json','{}')
    with pytest.raises(ValueError,match='重复'):read_input_package(buffer.getvalue())
