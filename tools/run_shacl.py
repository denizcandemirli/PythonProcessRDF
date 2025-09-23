import os
from rdflib import Graph


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

SHAPES_TTL = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix bot: <https://w3id.org/bot#> .
@prefix core: <https://spec.industrialontologies.org/ontology/core/Core/> .

[] a sh:NodeShape ;
   sh:targetClass bot:Space ;
   sh:property [
     sh:path bot:adjacentZone ;
     sh:minCount 0 ;
   ] .

[] a sh:NodeShape ;
   sh:property [
     sh:path core:hasFunction ;
     sh:minCount 0 ;
   ] .
"""


def main():
    # Minimal offline report (rdflib SHACL plugin not assumed). We serialize shapes and note not executed.
    p_shapes = os.path.join(ROOT, 'shapes_min.ttl')
    with open(p_shapes, 'w', encoding='utf-8') as f:
        f.write(SHAPES_TTL)
    with open(os.path.join(ROOT, 'shacl_report.md'), 'w', encoding='utf-8') as f:
        f.write('Minimal SHACL shapes written to shapes_min.ttl. No engine executed.\n')
        f.write('To run: install pyshacl and run `pyshacl -s shapes_min.ttl -m -f human *.rdf`.')
    print('WROTE shapes_min.ttl and shacl_report.md (instructions)')


if __name__ == '__main__':
    main()


