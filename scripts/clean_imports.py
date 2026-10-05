from pathlib import Path
import re
for p in Path('frontend/src/app/pages').glob('*.tsx'):
    s=p.read_text()
    body=re.sub(r'import[\s\S]*?from ["\'][^"\']+["\'];\n','',s)
    def tidy(m):
        names=[]
        for item in m.group(2).split(','):
            item=item.strip()
            name=item.split(' as ')[-1].strip()
            if name and re.search(r'\b'+re.escape(name)+r'\b',body): names.append(item)
        return f'import {m.group(1)}{{{", ".join(names)}}} from {m.group(3)};\n' if names else ''
    s=re.sub(r'import (type )?\{([\s\S]*?)\} from (["\'][^"\']+["\']);\n',lambda m:tidy(type('M',(),{'group':lambda self,n: (m.group(n) or '')})()),s)
    s=s.replace('import  {','import {')
    p.write_text(s)
