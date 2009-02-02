// Copyright (C) 2002-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Erik Svensson 2003.
// Modified by Garth N. Wells 2006-2008.
// Modified by Ola Skavhaug 2006.
// Modified by Magnus Vikstrom 2007.
// Modified by Niclas Jansson 2008.
//
// First added:  2002-12-03
// Last changed: 2009-01-14

#include <stdarg.h>
#include <boost/shared_ptr.hpp>

#include <dolfin/log/log.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/Array.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/main/MPI.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/graph/Graph.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/parameter/Parameter.h>
#include <dolfin/parameter/ParameterList.h>

#include "XMLObject.h"
#include "XMLVector.h"
#include "XMLMatrix.h"
#include "XMLMesh.h"
#include "XMLLocalMeshData.h"
#include "XMLMeshFunction.h"
#include "XMLDofMap.h"
#include "XMLFunction.h"
#include "XMLFiniteElement.h"
#include "XMLParameterList.h"
#include "XMLBLASFormData.h"
#include "XMLGraph.h"
#include "XMLFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFile::XMLFile(const std::string filename, bool gzip)
  : GenericFile(filename),
    header_written(false),
    mark(0),
    gzip(gzip)
{
  type = "XML";
  xmlObject = 0;
}
//-----------------------------------------------------------------------------
XMLFile::~XMLFile()
{
  if (xmlObject)
    delete xmlObject;
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (GenericVector& x)
{
  message(1, "Reading vector from file %s.", filename.c_str());

  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLVector(x);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (GenericMatrix& A)
{
  message(1, "Reading matrix from file %s.", filename.c_str());

  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLMatrix(A);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (Mesh& mesh)
{
  message(1, "Reading mesh from file %s.", filename.c_str());

  if (xmlObject)
    delete xmlObject;
  
  xmlObject = new XMLMesh(mesh);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (LocalMeshData& meshdata)
{
  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLLocalMeshData(meshdata);
  parseFile();

}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (MeshFunction<int>& meshfunction)
{
  message(1, "Reading int-valued mesh function from file %s.", filename.c_str());

  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLMeshFunction(meshfunction);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (MeshFunction<unsigned int>& meshfunction)
{
  message(1, "Reading uint-valued mesh function from file %s.", filename.c_str());

  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLMeshFunction(meshfunction);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (MeshFunction<double>& meshfunction)
{
  message(1, "Reading real-valued mesh function from file %s.", filename.c_str());

  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLMeshFunction(meshfunction);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (MeshFunction<bool>& meshfunction)
{
  message(1, "Reading bool-valued mesh function from file %s.", filename.c_str());

  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLMeshFunction(meshfunction);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (Function& v)
{
  // We are cheating here. Instead of actually parsing the XML for
  // Function data nested inside <function></function>, we just ignore
  // the nesting and look for the first occurence of the data which
  // might be outside of <function></function>.

  message(1, "Reading function from %s.", filename.c_str());

  // Read the mesh
  boost::shared_ptr<Mesh> mesh(new Mesh());
  *this >> *mesh;
  mesh->order();

  // Read the finite element specification
  std::string element_signature;
  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLFiniteElement(element_signature);
  parseFile(); 

  // Read the dof map specification
  std::string dofmap_signature;
  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLDofMap(dofmap_signature);
  parseFile(); 

  // Read the function (not really necessary, see comment above)
  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLFunction(v);
  parseFile(); 
  
  // Create Function
  boost::shared_ptr<FiniteElement> element(new FiniteElement(element_signature));
  boost::shared_ptr<DofMap> dofmap(new DofMap(dofmap_signature, *mesh));
  boost::shared_ptr<FunctionSpace> V(new FunctionSpace(mesh, element, dofmap));
  Function _v(V);

  // Read the vector
  *this >> _v.vector();

  // Assign function
  v = _v;

  v.rename("u", "discrete function from file data");
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (ParameterList& parameters)
{
  message(1, "Reading parameter list from file %s.", filename.c_str());

  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLParameterList(parameters);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (BLASFormData& blas)
{
  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLBLASFormData(blas);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (Graph& graph)
{
  message(1, "Reading graph from file %s.", filename.c_str());

  if (xmlObject)
    delete xmlObject;
  xmlObject = new XMLGraph(graph);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const GenericVector& x)
{
  // Open file
  FILE* fp = openFile();

  // Get vector values
  double* values = new double[x.size()];
  x.get(values);
  
  // Write vector in XML format
  fprintf(fp, "  <vector size=\"%u\"> \n", x.size() );
  for (unsigned int i = 0; i < x.size(); i++) 
  {
    fprintf(fp, "    <entry row=\"%u\" value=\"%.15g\"/>\n", i, values[i]);
    if ( i == (x.size() - 1))
      fprintf(fp, "  </vector>\n");
  }
  
  // Delete vector values
  delete [] values;

  // Close file
  closeFile(fp);
  
//  message(1, "Saved vector %s (%s) to file %s in DOLFIN XML format.", x.name().c_str(), x.label().c_str(), filename.c_str());
  message(1, "Saved vector  to file %s in DOLFIN XML format.", filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const GenericMatrix& A)
{
  // Open file
  FILE *fp = openFile();
  
  // Write matrix in XML format
  fprintf(fp, "  <matrix rows=\"%u\" columns=\"%u\">\n", A.size(0), A.size(1));
        
  Array<uint> columns;
  Array<double> values;

  for (unsigned int i = 0; i < A.size(0); i++)
  {
    A.getrow(i, columns, values);
    if (columns.size() > 0)
      fprintf(fp, "    <row row=\"%u\" size=\"%d\">\n", i, (int)columns.size());
    for (uint pos = 0; pos < columns.size(); pos++)
    {
      unsigned int j = columns[pos];
      double aij = values[pos];
      fprintf(fp, "      <entry column=\"%u\" value=\"%.15g\"/>\n", j, aij);
    }
    if (columns.size() > 0 )
      fprintf(fp, "    </row>\n");
  }
  fprintf(fp, "  </matrix>\n");

  // Close file
  closeFile(fp);

  message(1, "Saved matrix file %s in DOLFIN XML format.", filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const Mesh& mesh)
{
  // Open file
  FILE *fp = openFile();
  
  // Get cell type
  CellType::Type cell_type = mesh.type().cellType();

  // Write mesh in XML format
  fprintf(fp, "  <mesh celltype=\"%s\" dim=\"%u\">\n",
          CellType::type2string(cell_type).c_str(), mesh.geometry().dim());

  // Write vertices
  fprintf(fp, "    <vertices size=\"%u\">\n", mesh.numVertices());
  for(VertexIterator v(mesh); !v.end(); ++v)
  {
    Point p = v->point();

    switch ( mesh.geometry().dim() ) {
    case 1:
      fprintf(fp, "      <vertex index=\"%u\" x=\"%g\"/>\n",
              v->index(), p.x());
      break;
    case 2:
      fprintf(fp, "      <vertex index=\"%u\" x=\"%g\" y=\"%g\"/>\n",
              v->index(), p.x(), p.y());
      break;
    case 3:
      fprintf(fp, "      <vertex index=\"%u\" x=\"%g\" y=\"%g\" z=\"%g\" />\n",
              v->index(), p.x(), p.y(), p.z());
      break;
    default:
      error("The XML mesh file format only supports 1D, 2D and 3D meshes.");
    }
  }
  fprintf(fp, "    </vertices>\n");

  // Write cells
  fprintf(fp, "    <cells size=\"%u\">\n", mesh.numCells());
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    const uint* vertices = c->entities(0);
    dolfin_assert(vertices);

    switch ( cell_type )
    {
    case CellType::interval:
      fprintf(fp, "      <interval index=\"%u\" v0=\"%u\" v1=\"%u\"/>\n",
	      c->index(), vertices[0], vertices[1]);
      break;
    case CellType::triangle:
      fprintf(fp, "      <triangle index=\"%u\" v0=\"%u\" v1=\"%u\" v2=\"%u\"/>\n",
	      c->index(), vertices[0], vertices[1], vertices[2]);
      break;
    case CellType::tetrahedron:
      fprintf(fp, "      <tetrahedron index=\"%u\" v0=\"%u\" v1=\"%u\" v2=\"%u\" v3=\"%u\"/>\n",
              c->index(), vertices[0], vertices[1], vertices[2], vertices[3]);
      break;
    default:
      error("Unknown cell type: %u.", cell_type);
    }
  }
  fprintf(fp, "    </cells>\n");

  // Write mesh data
  const MeshData& data = mesh.data();
  if (data.meshfunctions.size() > 0 || data.arrays.size() > 0)
  {
    fprintf(fp, "    <data>\n");

    // Write mesh functions
    for (std::map<std::string, MeshFunction<uint>*>::const_iterator it = data.meshfunctions.begin();
         it != data.meshfunctions.end(); ++it)
    {
      fprintf(fp, "      <meshfunction name=\"%s\" type=\"uint\" dim=\"%d\" size=\"%d\">\n",
              it->first.c_str(), it->second->dim(), it->second->size());
      for (uint i = 0; i < it->second->size(); i++)
        fprintf(fp, "        <entity index=\"%d\" value=\"%d\"/>\n", i, it->second->get(i));
      fprintf(fp, "      </meshfunction>\n");
    }

    // Write arrays
    for (std::map<std::string, Array<uint>*>::const_iterator it = data.arrays.begin();
         it != data.arrays.end(); ++it)
    {
      fprintf(fp, "      <array name=\"%s\" type=\"uint\" size=\"%d\">\n",
              it->first.c_str(), (int) it->second->size());
      for (uint i = 0; i < it->second->size(); i++)
        fprintf(fp, "        <element index=\"%d\" value=\"%d\"/>\n", i, (*it->second)[i]);
      fprintf(fp, "      </array>\n");
    }
    fprintf(fp, "    </data>\n");
  }

  fprintf(fp, "  </mesh>\n");
 
  // Close file
  closeFile(fp);

  message(1, "Saved mesh to file %s in DOLFIN XML format.", filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const MeshFunction<int>& meshfunction)
{
  // Open file
  FILE *fp = openFile();
  
  // Write mesh in XML format
  fprintf(fp, "  <meshfunction type=\"int\" dim=\"%u\" size=\"%u\">\n",
          meshfunction.dim(), meshfunction.size());
  
  const Mesh& mesh = meshfunction.mesh();
  for(MeshEntityIterator e(mesh, meshfunction.dim()); !e.end(); ++e)
  {
      fprintf(fp, "    <entity index=\"%u\" value=\"%d\"/>\n",
              e->index(), meshfunction(*e));
  }

  fprintf(fp, "  </meshfunction>\n");
 
  // Close file
  closeFile(fp);
  
  message(1, "Saved mesh function to file %s in DOLFIN XML format.", filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const MeshFunction<unsigned int>& meshfunction)
{
  // Open file
  FILE *fp = openFile();
  
  // Write mesh in XML format
  fprintf(fp, "  <meshfunction type=\"uint\" dim=\"%u\" size=\"%u\">\n",
          meshfunction.dim(), meshfunction.size());
  
  const Mesh& mesh = meshfunction.mesh();
  for(MeshEntityIterator e(mesh, meshfunction.dim()); !e.end(); ++e)
  {
      fprintf(fp, "    <entity index=\"%u\" value=\"%d\"/>\n",
              e->index(), meshfunction(*e));
  }

  fprintf(fp, "  </meshfunction>\n");
 
  // Close file
  closeFile(fp);
  
  message(1, "Saved mesh function to file %s in DOLFIN XML format.", filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const MeshFunction<double>& meshfunction)
{
  // Open file
  FILE *fp = openFile();
  
  // Write mesh in XML format
  fprintf(fp, "  <meshfunction type=\"double\" dim=\"%u\" size=\"%u\">\n",
          meshfunction.dim(), meshfunction.size());

  const Mesh& mesh = meshfunction.mesh();
  for(MeshEntityIterator e(mesh, meshfunction.dim()); !e.end(); ++e)
  {
      fprintf(fp, "    <entity index=\"%u\" value=\"%g\"/>\n",
              e->index(), meshfunction(*e));
  }

  fprintf(fp, "  </meshfunction>\n");
 
  // Close file
  closeFile(fp);
  
  message(1, "Saved mesh function to file %s in DOLFIN XML format.", filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const MeshFunction<bool>& meshfunction)
{
  // Open file
  FILE *fp = openFile();
  
  // Write mesh in XML format
  fprintf(fp, "  <meshfunction type=\"bool\" dim=\"%u\" size=\"%u\">\n",
          meshfunction.dim(), meshfunction.size());

  const Mesh& mesh = meshfunction.mesh();
  std::string value;
  for (MeshEntityIterator e(mesh, meshfunction.dim()); !e.end(); ++e)
  {
    value = (meshfunction(*e) ? "true" : "false");
    fprintf(fp, "    <entity index=\"%u\" value=\"%s\"/>\n",
              e->index(), value.c_str());
  }

  fprintf(fp, "  </meshfunction>\n");
 
  // Close file
  closeFile(fp);
  
  message(1, "Saved mesh function to file %s in DOLFIN XML format.", filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const Function& v)
{
  // Begin function
  FILE *fp = openFile();
  fprintf(fp, "  <function> \n");
  closeFile(fp);

  // Write the mesh
  *this << v.function_space().mesh();
  
  // Write the vector
  *this << v.vector();

  // Write the finite element
  fp = openFile();
  fprintf(fp, "  <finiteelement signature=\"%s\"/>\n", v.function_space().element().signature().c_str());
  closeFile(fp);

  // Write the dof map
  fp = openFile();
  fprintf(fp, "  <dofmap signature=\"%s\"/>\n", v.function_space().dofmap().signature().c_str());
  closeFile(fp);

  // End function
  fp = openFile();
  fprintf(fp, "  </function> \n");
  closeFile(fp);

  message(1, "Saved function to file %s in DOLFIN XML format.", filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const ParameterList& parameters)
{
  // Open file
  FILE *fp = openFile();

  // Write parameter list in XML format
  fprintf(fp, "  <parameters>\n" );

  for (ParameterList::const_iterator it = parameters.parameters.begin(); it != parameters.parameters.end(); ++it)
  {
    const Parameter parameter = it->second;

    switch ( parameter.type() )
    {
    case Parameter::type_int:
      fprintf(fp, "    <parameter name=\"%s\" type=\"int\" value=\"%d\"/>\n",
	      it->first.c_str(), static_cast<int>(parameter));
      break;
    case Parameter::type_real:
      fprintf(fp, "    <parameter name=\"%s\" type=\"real\" value=\"%.16e\"/>\n",
	      it->first.c_str(), static_cast<double>(parameter));
      break;
    case Parameter::type_bool:
      if (static_cast<bool>(parameter))
	fprintf(fp, "    <parameter name=\"%s\" type=\"bool\" value=\"true\"/>\n",
		it->first.c_str());
      else
	fprintf(fp, "    <parameter name=\"%s\" type=\"bool\" value=\"false\"/>\n",
		it->first.c_str());
      break;
    case Parameter::type_string:
      fprintf(fp, "    <parameter name=\"%s\" type=\"string\" value=\"%s\"/>\n",
	      it->first.c_str(), static_cast<std::string>(parameter).c_str());
      break;
    default:
      ; // Do nothing
    }
  }
  
  fprintf(fp, "  </parameters>\n" );

  // Close file
  closeFile(fp);

  message(1, "Saved parameters to file %s in DOLFIN XML format.", filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const Graph& graph)
{
  // Open file
  FILE *fp = openFile();
  
  // Get graph type and number of vertices, edges and arches
  uint num_vertices = graph.numVertices();

  // Write graph in XML format
  fprintf(fp, "  <graph type=\"%s\">\n", graph.typestr().c_str());

  // Get connections (outgoing edges), offsets and weigts
  const uint* connections = graph.connectivity();
  const uint* offsets = graph.offsets();
  const uint* edge_weights = graph.edgeWeights();
  const uint* vertex_weights = graph.vertexWeights();

  dolfin_assert(connections);
  dolfin_assert(offsets);
  dolfin_assert(edge_weights);
  dolfin_assert(vertex_weights);
  
  // Write vertice header 
  fprintf(fp, "    <vertices size=\"%u\">\n", graph.numVertices());

  // Vertices
  for(uint i=0; i<num_vertices; ++i)
  {
	  fprintf(fp, 
          "      <vertex index=\"%u\" num_edges=\"%u\" weight=\"%u\"/>\n", i,
          graph.numEdges(i), vertex_weights[i]);
	  
  }
  fprintf(fp, "    </vertices>\n");

  fprintf(fp, "    <edges size=\"%u\">\n", graph.numEdges());
  // Edges
  for(uint i=0; i<num_vertices; ++i)
  {
    for(uint j=offsets[i]; j<offsets[i] + graph.numEdges(i); ++j)
    {
      // In undirected graphs an edge (v1, v2) is the same as edge (v2, v1)
      // and should not be stored twice
      if (graph.type() == Graph::directed || i < connections[j])
        fprintf(fp, 
        "      <edge v1=\"%u\" v2=\"%u\" weight=\"%u\"/>\n",
        i, connections[j], edge_weights[j]);
    }
  }
  fprintf(fp, "    </edges>\n");
  fprintf(fp, "  </graph>\n");
  
  // Close file
  closeFile(fp);

  message(1, "Saved graph to file %s in DOLFIN XML format.", filename.c_str());
}
//-----------------------------------------------------------------------------
FILE* XMLFile::openFile()
{
  // Cannot write gzipped files (yet)
  if (gzip)
    error("Unable to write data to file, gzipped XML (xml.gz) not supported for output.");

  // Open file
  FILE *fp = fopen(filename.c_str(), "r+");
  if (!fp)
    error("Unable to open file %s", filename.c_str());

  // Step to position before previously written footer
  //printf("Stepping to position: %ld\n", mark);
  fseek(fp, mark, SEEK_SET);
  fflush(fp);
  
  // Write DOLFIN XML format header
  if (!header_written)
  {
    fprintf(fp, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n" );
    fprintf(fp, "<dolfin xmlns:dolfin=\"http://www.fenics.org/dolfin/\">\n" );
    
    header_written = true;
  }

  return fp;
}
//-----------------------------------------------------------------------------
void XMLFile::closeFile(FILE* fp)
{
  // Get position in file before writing footer
  mark = ftell(fp);
  //printf("Position in file before writing footer: %ld\n", mark);

  // Write DOLFIN XML format footer
  if (header_written)
    fprintf(fp, "</dolfin>\n");

  // Close file
  fclose(fp);
}
//-----------------------------------------------------------------------------
void XMLFile::parseFile()
{
  // Notify that file is being opened
  xmlObject->open(filename);

  // Parse file using the SAX interface
  parseSAX();
  
  // Notify that file is being closed
  if (!xmlObject->close())
    error("Unable to find data in XML file.");
}
//-----------------------------------------------------------------------------
void XMLFile::parseSAX()
{
  // Set up the sax handler. Note that it is important that we initialise
  // all (24) fields, even the ones we don't use!
  xmlSAXHandler sax = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  
  // Set up handlers for parser events
  sax.startDocument = sax_start_document;
  sax.endDocument   = sax_end_document;
  sax.startElement  = sax_start_element;
  sax.endElement    = sax_end_element;
  sax.warning       = sax_warning;
  sax.error         = sax_error;
  sax.fatalError    = sax_fatal_error;
  
  // Parse file
  xmlSAXUserParseFile(&sax, (void *) xmlObject, filename.c_str());
}
//-----------------------------------------------------------------------------
// Callback functions for the SAX interface
//-----------------------------------------------------------------------------
void dolfin::sax_start_document(void *ctx)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void dolfin::sax_end_document(void *ctx)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void dolfin::sax_start_element(void *ctx,
			       const xmlChar *name, const xmlChar **attrs)
{
  ( (XMLObject *) ctx )->startElement(name, attrs);
}
//-----------------------------------------------------------------------------
void dolfin::sax_end_element(void *ctx, const xmlChar *name)
{
  ( (XMLObject *) ctx )->endElement(name);
}
//-----------------------------------------------------------------------------
void dolfin::sax_warning(void *ctx, const char *msg, ...)
{
  va_list args;
  va_start(args, msg);
  char buffer[DOLFIN_LINELENGTH];
  vsnprintf(buffer, DOLFIN_LINELENGTH, msg, args);
  warning("Incomplete XML data: " + std::string(buffer));
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::sax_error(void *ctx, const char *msg, ...)
{
  va_list args;
  va_start(args, msg);
  char buffer[DOLFIN_LINELENGTH];
  vsnprintf(buffer, DOLFIN_LINELENGTH, msg, args);
  error("Illegal XML data: " + std::string(buffer));
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::sax_fatal_error(void *ctx, const char *msg, ...)
{
  va_list args;
  va_start(args, msg);
  char buffer[DOLFIN_LINELENGTH];
  vsnprintf(buffer, DOLFIN_LINELENGTH, msg, args);
  error("Illegal XML data: " + std::string(buffer));
  va_end(args);
}
//-----------------------------------------------------------------------------
