"""Reproduce the S2-era source audit using copied source files and saved country lookup."""

# Discover the configuration module; data locations come from config.json.
import sys as _path_sys
from pathlib import Path as _ConfigPath
_path_root = _ConfigPath(__file__).resolve().parent
while not (_path_root / "example_config.json").is_file() and _path_root != _path_root.parent:
    _path_root = _path_root.parent
_path_sys.path.insert(0, str(_path_root))
import config as _path_config
import csv,json,math,collections,datetime
DC = _path_config.get_path("DATA_COLLECTION_DIR")
OUT=DC
NAMES={'Norway':'NO','Sweden':'SE','Denmark':'DK','United Kingdom':'GB','GB':'GB'}
rows=[];inventory=[];rejected=collections.Counter()
for src,base in [('gbif','kelp_gbif_europe'),('obis','kelp_obis_europe'),('jncc','kelp_jncc')]:
 for status in ['present','absent']:
  path=_path_config.get_path(src.upper()+'_'+status.upper());raw=list(csv.DictReader(path.open()))
  inventory.append({'file':src+'/'+path.name,'rows':len(raw)})
  for rr in raw:
   r=dict(rr);r['source']=src;r['occurrenceStatus']=status.upper()
   r['species_set']={s.strip().lower().replace('laminaria saccharina','saccharina latissima') for s in (r.get('species') or r.get('scientificName') or '').split(',') if s.strip()}
   try:
    lat,lon=float(r['decimalLatitude']),float(r['decimalLongitude'])
    assert math.isfinite(lat) and math.isfinite(lon) and -90<=lat<=90 and -180<=lon<=180 and (lat,lon)!=(0,0)
   except (ValueError,KeyError,AssertionError,TypeError):rejected[src+':invalid_coords']+=1;continue
   date=(r.get('eventDate') or '').strip()[:10]
   try:datetime.date.fromisoformat(date)
   except ValueError:rejected[src+':invalid_day_date']+=1;continue
   r['date']=date;r['lat']=lat;r['lon']=lon;r['site']=(round(lat,4),round(lon,4));r['event']=(*r['site'],date)
   r['country_code']=(r.get('countryCode') or NAMES.get(r.get('country'), '') or ('GB' if src=='jncc' else '')).upper()
   r['country_method']='source_country' if src!='jncc' else 'JNCC_UK_source'
   rows.append(r)

all_dated_rows=rows
rows=[r for r in rows if r['date']>='2016-01-01']
lookup={tuple(x['site']):(x['country'],x['coast_distance_km']) for x in json.loads((OUT/'observation_audit_country_lookup.json').read_text())}
def geo(lat,lon):
 return lookup[(round(lat,4),round(lon,4))]
# Coordinate lookup was generated with Natural Earth admin-0 countries v5.1.1,
# nearest land geometry for coastal points. Rebuild lookup if input sites change.
# Explicit country fields first; recover missing fields from GBIF/OBIS site countries only.
country_sites=collections.defaultdict(set)
for r in rows:
 if r['country_method']=='source_country' and r['country_code']:country_sites[tuple(r['site'])].add(r['country_code'])
geo_cache={};fixes=[]
for r in rows:
 if r['country_method']!='source_country' or not r['country_code']:
  cs=country_sites[tuple(r['site'])]
  if len(cs)==1:r['country_code']=next(iter(cs));r['country_method']='matched_source_site'
  else:
   site=tuple(r['site'])
   if site not in geo_cache:geo_cache[site]=geo(r['lat'],r['lon'])
   r['country_code'],r['coast_distance_km']=geo_cache[site];r['country_method']='coordinate_country_or_nearest_coast'
 # Two OBIS records encode Georgia for Kiel, Germany; explicit documented correction.
 if r['country_code']=='GE' and 54<r['lat']<55 and 10<r['lon']<11:
  fixes.append({'id':r.get('obisID'),'original':'GE','corrected':'DE','lat':r['lat'],'lon':r['lon']});r['country_code']='DE';r['country_method']='corrected_country_using_coordinates'
 r['species_set']=set(r['species_set'])
def uncertainty(r):
 raw=(r.get('coordinateUncertaintyInMeters') or '').strip()
 if not raw:return None
 try:
  u=float(raw);return u if math.isfinite(u) and u>=0 else None
 except ValueError:return None

def dedup(rr):
 groups=collections.defaultdict(list)
 for r in rr:groups[(r['occurrenceStatus'],*r['event'])].append(r)
 result=[]
 for key,members in groups.items():
  members.sort(key=lambda r:({'gbif':0,'obis':1,'jncc':2}[r['source']],str(r.get('gbifID') or r.get('obisID') or r.get('sample_ref'))))
  r=dict(members[0]);r['species_set']=sorted(set.union(*(x['species_set'] for x in members)))
  r['sources']=sorted({x['source'] for x in members});r['n_source_rows']=len(members)
  r['source_record_ids']=sorted({record_id(x) for x in members})
  r['source_dataset_keys']=sorted({str(x.get('datasetKey') or '') for x in members if x.get('datasetKey')})
  reported=[uncertainty(x) for x in members if uncertainty(x) is not None]
  r['best_reported_uncertainty']=min(reported) if reported else ''
  r['known_le10']=any(uncertainty(x) is not None and uncertainty(x)<=10 for x in members)
  r['sugar_known_le10']=any('saccharina latissima' in x['species_set'] and uncertainty(x) is not None and uncertainty(x)<=10 for x in members)
  result.append(r)
 return result
# Spatial/temporal matching uses original coordinates, spherical great-circle
# distance and inclusive thresholds. ECEF cells only accelerate exact checks.
DISTANCE_M=10.0
DAYS=5
R_EARTH=6371008.8

def prep(r):
 r=dict(r)
 r['ordinal']=datetime.date.fromisoformat(r['date']).toordinal()
 a,b=math.radians(r['lat']),math.radians(r['lon'])
 xyz=(R_EARTH*math.cos(a)*math.cos(b),R_EARTH*math.cos(a)*math.sin(b),R_EARTH*math.sin(a))
 r['cell']=tuple(math.floor(v/DISTANCE_M) for v in xyz)
 return r

def distance(a,b):
 la,lb=math.radians(a['lat']),math.radians(b['lat'])
 dlat=lb-la;dlon=math.radians(b['lon']-a['lon'])
 h=math.sin(dlat/2)**2+math.cos(la)*math.cos(lb)*math.sin(dlon/2)**2
 return 2*R_EARTH*math.asin(min(1,math.sqrt(h)))

def neighbors(r,index):
 x,y,z=r['cell']
 for dx in [-1,0,1]:
  for dy in [-1,0,1]:
   for dz in [-1,0,1]:
    yield from index.get((x+dx,y+dy,z+dz),())

def record_id(r):
 return r['source']+':'+str(r.get('gbifID') or r.get('obisID') or r.get('sample_ref'))

# All known PRESENT evidence, including low-precision observations and the
# five days before the S2 reporting cutoff, participates in the absence veto.
all_presence=[prep(r) for r in all_dated_rows if r['occurrenceStatus']=='PRESENT' and r['date']>='2015-12-27']
# Broaden the veto to every PRESENT kelp record in the GBIF master, including
# genus-level identifications and species outside the classifier target list.
for r0 in csv.DictReader((_path_config.get_path("GBIF_MASTER")).open()):
 if r0.get('occurrenceStatus')!='PRESENT':continue
 date=(r0.get('eventDate') or '').strip()[:10]
 try:
  datetime.date.fromisoformat(date)
  lat,lon=float(r0['decimalLatitude']),float(r0['decimalLongitude'])
  assert math.isfinite(lat) and math.isfinite(lon) and -90<=lat<=90 and -180<=lon<=180 and (lat,lon)!=(0,0)
 except (ValueError,KeyError,TypeError,AssertionError):continue
 if date<'2015-12-27':continue
 r=dict(r0);r.update(source='gbif',date=date,lat=lat,lon=lon,species_set={(r0.get('species') or r0.get('scientificName') or r0.get('genus') or 'unresolved kelp').lower()})
 all_presence.append(prep(r))
presence_index=collections.defaultdict(list)
for r in all_presence:presence_index[r['cell']].append(r)

INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
def is_inaturalist(r):
 return (str(r.get('datasetKey') or '').strip().lower()==INAT_DATASET_KEY
         or any('inaturalist' in str(r.get(k) or '').lower()
                for k in ['datasetName','publisher','publishingOrganization','occurrenceID']))
# Exclude iNaturalist as candidate data BEFORE deduplication. Keep its presence
# evidence in presence_index: a discarded training record can still veto absence.
candidate_rows=[r for r in rows if not is_inaturalist(r)]
inat_excluded=collections.Counter(r['occurrenceStatus'] for r in rows if is_inaturalist(r))

# Read the current PRESENT allowlist. Selection thresholds are defined below.
sesimbra_targets=set()
in_list=False
for line in (_path_config.get_path("SPECIES_POLICY")).read_text().splitlines():
 if line.strip()=='species_present:':in_list=True;continue
 if in_list:
  if line.strip().startswith('- '):sesimbra_targets.add(line.strip()[2:].lower())
  elif line and not line[0].isspace() and not line.startswith('#'):break
assert len(sesimbra_targets)==8

# Deduplication and absence exclusion are separate operations. Date-tolerance
# matching never merges distinct PRESENT visits or removes PRESENT records.
selected_present=[r for r in candidate_rows if r['occurrenceStatus']=='PRESENT' and (uncertainty(r) is None or uncertainty(r)<=10)]
present=dedup(selected_present)
def source_stage_counts(records):
 keys={(r['occurrenceStatus'],*r['event']) for r in records}
 return dict(collections.Counter(key[0] for key in keys))
print('Valid S2-era unique observations:',source_stage_counts(rows))
print('After iNaturalist exclusion:',source_stage_counts(candidate_rows))
print('After PRESENT uncertainty selection:',source_stage_counts(selected_present+[r for r in candidate_rows if r['occurrenceStatus']=='ABSENT']))
sesimbra=dedup([r for r in selected_present if r['species_set'] & sesimbra_targets])
absent_groups=collections.defaultdict(list)
for r in candidate_rows:
 if r['occurrenceStatus']=='ABSENT':absent_groups[(r['occurrenceStatus'],*r['event'])].append(r)
retained_members=[];exclusions=[]
for key,members in absent_groups.items():
 witness=None
 for raw in members:
  r=prep(raw)
  for p in neighbors(r,presence_index):
   days=abs(r['ordinal']-p['ordinal'])
   if days>DAYS:continue
   meters=distance(r,p)
   if meters<=DISTANCE_M:
    witness={'absent_key':list(key),'absent_id':record_id(r),'present_id':record_id(p),'present_species':sorted(p['species_set']),'days_difference':days,'distance_m':meters};break
  if witness:break
 if witness:exclusions.append(witness)
 else:retained_members.extend(members)
absent=dedup(retained_members)

def stats(cc,pp):
 ps=[r for r in pp if r['country_code']==cc]
 return [len(ps),sum(r['known_le10'] for r in ps),sum(not r['known_le10'] for r in ps),sum(r['country_code']==cc for r in absent)]

def write_table(name,header,table):
 with (DC/name).open('w',newline='') as f:
  w=csv.writer(f);w.writerow(header);w.writerows(table)
 # Read-back checks: complete rows and integer counts, including totals.
 saved=list(csv.reader((DC/name).open()))
 assert saved[0]==header and len(saved)==len(table)+1
 assert all(len(r)==len(header) and all(v.isdigit() for v in r[1:]) for r in saved[1:])

order=[('GB','United Kingdom'),('FR','France'),('IE','Ireland'),('IM','Isle of Man'),('JE','Jersey'),('IS','Iceland'),('DE','Germany'),('SJ','Svalbard and Jan Mayen'),('GG','Guernsey'),('RU','Russia'),('ES','Spain'),('FO','Faroe Islands'),('NL','Netherlands'),('PT','Portugal'),('MA','Morocco')]
# Sesimbra summary covers every country represented by either selected positives
# or the shared negative set. A multi-species event contributes only once.
country_names=dict(order+[('NO','Norway'),('SE','Sweden'),('DK','Denmark'),('BE','Belgium'),('FI','Finland'),('EE','Estonia'),('IT','Italy'),('PL','Poland'),('AX','Åland Islands')])
sesimbra_country_codes={r['country_code'] for r in sesimbra+absent}
assert sesimbra_country_codes <= country_names.keys(), sesimbra_country_codes-country_names.keys()
sesimbra_table=[[country_names[cc],*stats(cc,sesimbra)] for cc in sorted(sesimbra_country_codes,key=lambda c:country_names[c])]
sesimbra_table.append(['All locations combined']+[sum(r[i] for r in sesimbra_table) for i in range(1,5)])
write_table('s2_sesimbra_like_species_summary.csv',['Country / territory','PRESENT, Sesimbra-like species','Precisions <=10m','Unknown precision only','ABSENT after exclusion'],sesimbra_table)
assert all(r[1]==r[2]+r[3] for r in sesimbra_table)

# Validation of policy invariants and country rollups.
assert not any(is_inaturalist(r) for r in selected_present+retained_members)
for collection in [present,sesimbra,absent]:
 keys=[(r['occurrenceStatus'],*r['event']) for r in collection]
 assert len(keys)==len(set(keys))
assert len(absent)+len(exclusions)==len(absent_groups)
summary={'date':'2026-10-08','dedup':'status + round(latitude,4) + round(longitude,4) + exact calendar date; no temporal or distance clustering','absence_filter':'any source member has PRESENT of any kelp within <=10m great-circle distance and <=5 calendar days; all available presence precisions used, including GBIF master','present_total':len(present),'sesimbra_present_total':len(sesimbra),'sesimbra_species':sorted(sesimbra_targets),'sesimbra_species_counts':{sp:len(dedup([r for r in selected_present if sp in r['species_set']])) for sp in sorted(sesimbra_targets)},'iNaturalist_excluded_source_rows':dict(inat_excluded),'absent_before_exclusion':len(absent_groups),'absent_excluded':len(exclusions),'absent_retained':len(absent),'absent_by_country':dict(collections.Counter(r['country_code'] for r in absent)),'verification':'iNaturalist excluded from candidates; unique keys; partition and country totals reconciled','exclusion_evidence':exclusions}
(DC/'observation_summary_audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='exclusion_evidence'},indent=2))

# Observation-level export for the Sesimbra experiment.
# Keep canonical pipeline columns, stable dedup-key identifiers and provenance.
import hashlib
observation_fields=[
 'observation_id','gbifID','decimalLatitude','decimalLongitude','eventDate',
 'species','occurrenceStatus','source','coordinateUncertaintyInMeters',
 'country_code','country_method','sources','source_record_ids',
 'source_dataset_keys','source_row_count','any_record_precision_le10m',
 'best_reported_coordinate_uncertainty_m','absence_check_passed',
]
def export_observations(name,records):
 records=sorted(records,key=lambda r:(r['country_code'],r['occurrenceStatus'],r['date'],r['lat'],r['lon']))
 exported=[]
 for r in records:
  key=(r['occurrenceStatus'],*r['event'])
  stable_id=hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()
  native_id=str(r.get('gbifID') or r.get('obisID') or r.get('sample_ref'))
  fetch_id=native_id if r['source']=='gbif' else r['source']+':'+native_id
  exported.append({
   'observation_id':stable_id,'gbifID':fetch_id,
   'decimalLatitude':r['decimalLatitude'],'decimalLongitude':r['decimalLongitude'],
   'eventDate':r['date'],'species':', '.join(r['species_set']),
   'occurrenceStatus':r['occurrenceStatus'],'source':r['source'],
   'coordinateUncertaintyInMeters':r.get('coordinateUncertaintyInMeters',''),
   'country_code':r['country_code'],'country_method':r['country_method'],
   'sources':';'.join(r['sources']),'source_record_ids':';'.join(r['source_record_ids']),
   'source_dataset_keys':';'.join(r['source_dataset_keys']),
   'source_row_count':r['n_source_rows'],'any_record_precision_le10m':r['known_le10'],
   'best_reported_coordinate_uncertainty_m':r['best_reported_uncertainty'],
   'absence_check_passed':True if r['occurrenceStatus']=='ABSENT' else '',
  })
 with (DC/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=observation_fields);w.writeheader();w.writerows(exported)
 saved=list(csv.DictReader((DC/name).open()))
 assert len(saved)==len(records)==len({r['observation_id'] for r in saved})
 assert not any(INAT_DATASET_KEY in r['source_dataset_keys'] for r in saved)
 return saved

observation_exports={
 's2_sesimbra_like_observations.csv':export_observations('s2_sesimbra_like_observations.csv',sesimbra+absent),
}
assert len(observation_exports['s2_sesimbra_like_observations.csv'])==sesimbra_table[-1][1]+sesimbra_table[-1][4]
summary['observation_exports']={name:{'rows':len(rs),'present':sum(r['occurrenceStatus']=='PRESENT' for r in rs),'absent':sum(r['occurrenceStatus']=='ABSENT' for r in rs)} for name,rs in observation_exports.items()}
(DC/'observation_summary_audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['observation_exports'],indent=2))
