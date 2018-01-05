#include <dolfin.h>
#include <boost/timer/timer.hpp>

using namespace dolfin;

int main()
{
  //std::array<Point, 2> pt = {Point(0.,0.), Point(1.,1.)};
  //auto mesh = std::make_shared<Mesh>(RectangleMesh::create(MPI_COMM_WORLD, pt, {{320, 320}}, CellType::Type::triangle));
  std::array<Point, 2> pt = {Point(0.,0.,0.), Point(1.,1.,1.)};
  auto mesh = std::make_shared<Mesh>(BoxMesh::create(MPI_COMM_WORLD, pt,
                                                     {{200, 200, 200}}, CellType::Type::tetrahedron));

  {
    Timer t0("old iterators");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    for (CellIterator c(*mesh); !c.end(); ++c)
    {
      for (VertexIterator v(*c); !v.end(); ++v)
        p += v->index();
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }

  {
    Timer t0("new (auto)");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    for (auto &c : cells(*mesh))
    {
      for (auto &v : vertices(c))
        p += v.index();
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }

  {
    Timer t0("new (no auto)");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    for (auto &c : cells(*mesh))
    {
      vertices vert(c);
      const auto& v0 = vert.begin();
      const auto& v1 = vert.end();
      for (auto v = v0; v != v1; ++v)
        p += v->index();
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }

  {
    Timer t0("new (outer) + old (inner)");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    for (auto &c : cells(*mesh))
    {
      // Use old iterator
      for (VertexIterator v(c); !v.end(); ++v)
        p += v->index();
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }


  list_timings(TimingClear::clear, {{TimingType::wall}});
  return 0;
}
