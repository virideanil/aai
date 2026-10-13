"""Authored compositional English/math supervision with disjoint structures."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from nanoai.meaning import render, parse_checked_english, separate_meanings
from nanoai.core import digest_bytes, file_digest
from nanoai.memory import canonical


def build():
    rows=[]
    training_symbols={'x':'the door is open','y':'the lamp is lit','z':'the pump is running'}
    forms=[
        ('leaf','x','The door is open.'),
        ('negate',{'not':'y'},'The lamp is not lit.'),
        ('conjunction',{'and':['x','y']},'The door is open and the lamp is lit.'),
        ('inclusive',{'or':['x','z']},'At least one condition holds: the door is open, or the pump is running. Both are allowed.'),
        ('exclusive',{'xor':['y','z']},'Exactly one of the lamp being lit and the pump running is true.'),
        ('equivalence',{'eq':['x','y']},'The door is open if and only if the lamp is lit.'),
        ('negative-conjunct',{'and':[{'not':'x'},'z']},'The door is not open, and the pump is running.'),
        ('negative-exclusive',{'xor':[{'not':'y'},'x']},'Exactly one holds: the lamp is not lit; the door is open.'),
        ('ambiguous-separating',{'alternatives':[{'or':['x',{'and':['y','z']}]},{'and':[{'or':['x','y']},'z']}]},
         'The note says: the door is open or the lamp is lit and the pump is running. Grouping is unspecified. Preserve both readings. Current truth values x=1,y=1,z=0 are an execution example, not a disambiguation.'),
        ('ambiguous-agreement',{'alternatives':[{'or':['x',{'and':['y','z']}]},{'and':[{'or':['x','y']},'z']}]},
         'The note says: the door is open or the lamp is lit and the pump is running. Grouping is unspecified. Preserve both readings. Current truth values x=1,y=1,z=1 are an execution example, not a disambiguation.'),
    ]
    validation=[
        ('negate',{'not':'x'},'The door is not open.'),
        ('inclusive',{'or':['y','z']},'The lamp is lit or the pump is running, possibly both.'),
        ('ambiguous',{'alternatives':[{'or':['y',{'and':['x','z']}]},{'and':[{'or':['y','x']},'z']}]},
         'The lamp is lit or the door is open and the pump is running. The grouping is unresolved. Keep both interpretations.'),
    ]
    confirmation_symbols={'x':'the alarm is active','y':'the valve is open','z':'the fan is spinning'}
    confirmation=[
        ('outer-negation',{'not':{'and':['x','y']}},'It is not true that both the alarm is active and the valve is open.'),
        ('nested-composition',{'or':['z',{'and':['x','y']}]},'The fan is spinning, or else both the alarm is active and the valve is open. These alternatives may both hold.'),
        ('repeated-referent',{'xor':['x','x']},'Exactly one of these two identical conditions holds: the alarm is active; the alarm is active.'),
        ('ambiguous-scope',{'alternatives':[{'not':{'and':['x','y']}},{'and':[{'not':'x'},'y']}]},
         'The note reads: not the alarm is active and the valve is open. The author has not specified whether not covers the first condition or the entire conjunction. Retain both readings.'),
    ]
    for split,examples,symbols in [('train',forms,training_symbols),('validation',validation,training_symbols),('confirmation',confirmation,confirmation_symbols)]:
        for name,expression,english in examples:
            canonical_english=render(expression,symbols)
            if separate_meanings(expression,parse_checked_english(canonical_english,symbols),symbols) is not None:
                raise ValueError('Teaching pair fails exact roundtrip')
            for direction in ('english_to_math','math_to_english'):
                forward=direction=='english_to_math'
                question=('Express the statement as a JSON Boolean formula. Use the string variable names and objects with not, and, or, xor or eq. Binary operators take two-element lists. For unresolved readings use {"alternatives":[formula1,formula2]}. Return only JSON.' if forward else
                    'Express the formula in checked English. Use the exact condition phrases. Grammar: not (A); both (A) and (B); at least one of (A) and (B); exactly one of (A) and (B); equivalent (A) and (B). For alternative interpretations use readings (A) or (B). Return only the expression.')
                observations=['Conditions: '+json.dumps(symbols,separators=(',',':')),
                              ('Statement: '+english if forward else 'Formula: '+json.dumps(expression,separators=(',',':')))]
                rows.append({'id':split+'-'+name+'-'+('en-math' if forward else 'math-en'),
                    'split':split,'family':name,'direction':direction,'question':question,
                    'observations':observations,'target':json.dumps(expression,separators=(',',':')) if forward else canonical_english,
                    'criteria':{'direction':direction,'symbols':symbols,'expression':expression,
                                'assignment':{'x':1,'y':1,'z':1 if name=='ambiguous-agreement' else 0}},
                    'provenance':'Authored finite semantics; natural English annotated explicitly; reverse checking grammar verified exhaustively in the declared context.'})
    # Replay is separately authored. These examples are not confirmation labels.
    replay=[
        ('english','Answer in one short sentence.','What is 3 plus 4?','3 plus 4 is 7.'),
        ('turkish','Translate into Turkish. Return only the translation.','The door is open.','Kapı açık.'),
        ('conduct','Return only one word: verified or unverified.','The test has been written but has not been run.','unverified'),
        ('privacy','Return only one word: local or public.','The private source must remain on this computer.','local'),
    ]
    for index in range(5):
        for name,question,observation,target in replay:
            rows.append({'id':f'replay-{index}-{name}','split':'replay','family':name,'direction':'replay',
                         'question':question,'observations':[observation],'target':target,
                         'provenance':'Four distinct authored language/conduct replay examples, sampled five times to balance the 20 semantic teaching directions by example count; duplication adds no evidence. Token counts are reported separately.'})
    controls=[
        ('arithmetic','Return only the integer answer.','What is 7 plus 8?','15',{'kind':'exact','accepted':['15']}),
        ('turkish','Translate into Turkish. Return only the translation.','The door is closed.','Kapı kapalı.',{'kind':'contains','required':['kapı','kapalı'],'forbidden':['açık']}),
        ('conduct','Return only one word: verified or unverified.','The build has never been executed.','unverified',{'kind':'exact','accepted':['unverified']}),
        ('ambiguity','Return only one word: clear or ambiguous.','The pilot told the observer that she was late. Which person does she refer to?','ambiguous',{'kind':'exact','accepted':['ambiguous']}),
    ]
    for name,q,o,t,checks in controls:
        rows.append({'id':'control-'+name,'split':'control','family':name,'direction':'control',
            'question':q,'observations':[o],'target':t,'criteria':checks,
            'provenance':'Fresh authored scope/language/conduct check; separate from teaching and validation.'})
    explanations={
        'leaf':'The statement names one supplied condition. Preserve its symbol without adding an operator.',
        'negate':'The statement denies the supplied condition. Apply not to that condition alone.',
        'conjunction':'Both named conditions must hold. Join those conditions with and, preserving each referent.',
        'inclusive':'At least one condition must hold and both are allowed. Use inclusive or.',
        'exclusive':'Exactly one condition must hold. Use xor; ordinary or would also allow both.',
        'equivalence':'The two conditions must agree in truth value. Use eq for the stated if-and-only-if relation.',
        'negative-conjunct':'Only the first condition is negated. Conjoin its negation with the positive second condition.',
        'negative-exclusive':'Negation applies to the first listed condition. Exclusive choice then compares that negation with the other condition.',
        'ambiguous-separating':'Grouping is unresolved. Retain both the or-of-and and the and-of-or interpretations. One execution assignment can separate readings but cannot establish the author intent.',
        'ambiguous-agreement':'Grouping is unresolved. Both candidate readings happen to agree on the supplied assignment, which does not make their general meanings identical. Retain both readings.',
        'ambiguous':'The ungrouped and/or statement has two allowed readings. Preserve both operator scopes in alternatives.',
        'english':'Three plus four equals seven. Express that sum in the requested short English sentence.',
        'turkish':'The door is open translates to Kapı açık. Preserve the positive state of the door.',
        'conduct':'A written test has not yet produced an execution result. Its outcome is unverified.',
        'privacy':'The source is explicitly private and must stay on this computer. Its permitted scope is local.',
    }
    for row in rows:
        if row['split'] in {'train','validation','replay'}:
            row['teaching_reason']=explanations[row['family']]
            if row['direction']=='math_to_english':
                row['teaching_reason'] += ' Render the same operator tree in the declared English grammar with the exact supplied condition phrases.'
            row['reason_provenance']='Authored explanation of the declared finite semantics, checked with the source annotation and exact roundtrip.'
    return {'schema':'nanoai.meaning-corpus.v1','scope':'Finite compositional predicate meaning, both directions; no universal English semantics claim.',
            'authoring_source':'user:msg_01a13684-b299-7d31-a12a-f9d656835f8d',
            'builder':file_digest(__file__),'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Preserve the previous frozen corpus')
    data=build();data['identity']=digest_bytes(canonical(data))
    args.out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'identity':data['identity'],'rows':len(data['rows'])}))
