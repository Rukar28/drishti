from pathlib import Path
p=Path('frontend/src/app/App.tsx');s=p.read_text();imports=s[:s.index('function Entry')];names=['Entry','Layout','Home','Assist','Navigation','Safety','History','Overview','Settings'];folder=Path('frontend/src/app/pages');folder.mkdir(exist_ok=True)
# Each screen stays independent; shared services and components are imported centrally.
for i,name in enumerate(names):
    start=s.index('function '+name+'(')
    end=s.index('function '+names[i+1]+'(') if i+1<len(names) else s.index('export default function App')
    body=s[start:end]
    if name=='Home':
        feature_start=body.index('const features =')
        features=body[feature_start:]
        body=body[:feature_start]
    if name=='Assist': body=features+body
    module=imports.replace('"./','"../') + ('import History from "./History";\n' if name=='Overview' else '') + body.replace('function '+name+'(', 'export default function '+name+'(',1)
    (folder/(name+'.tsx')).write_text(module)
app=s[s.index('export default function App'):]
new='import {useEffect} from "react";\nimport {Routes, Route, Navigate} from "react-router-dom";\nimport {startRealtime} from "./realtime";\nimport {DeviceStatus} from "./components";\n'+''.join(f'import {name} from "./pages/{name}";\n' for name in names)+app
p.write_text(new)
p=Path('frontend/src/app/realtime.ts');s=p.read_text();s=s.replace('!["WORLD_UPDATE", "DETECTION", "SENSOR_UPDATE"].includes(e.type)', '["SOS_ALERT", "MODE_CHANGED", "NAVIGATION_UPDATE", "ERROR", "HAZARD", "VOICE_COMMAND", "VOICE_TRANSCRIPT", "SYSTEM_STATUS"].includes(e.type)');p.write_text(s)
# Restrict Tailwind scanning to current source, excluding preserved drafts.
p=Path('frontend/tailwind.config.js');s=p.read_text().replace('./src/**/*.{js,ts,jsx,tsx}', './src/**/*.{js,ts,jsx,tsx}');p.write_text(s)
