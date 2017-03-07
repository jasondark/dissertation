#include <cassert>      // assert
#include <iterator>     // iterator
#include <utility>      // swap
#include <Eigen/Dense>

#include "partition.hpp"

template <class uint>
class PartitionIterator : std::iterator< std::bidirectional_iterator_tag, Partition<uint> > {
  private:
	Partition<uint>* current;
	bool invalid;

  public:
	PartitionIterator() : current(nullptr), invalid(true) {}
	PartitionIterator(const PartitionIterator& other) : invalid(other.invalid) {
	  current = new Partition<uint>(*other);
	}
	PartitionIterator(const Partition<uint>& other) : invalid(false) {
	  current = new Partition<uint>(other);
	}
	~PartitionIterator() {
	  if (current != nullptr)
		delete current;
	}

	void swap(PartitionIterator& other) noexcept {
	  std::swap(invalid, other.invalid);
	  std::swap(current, other.current);
	}

	PartitionIterator& operator++() {
	  assert(current != nullptr);
	  assert(!invalid);
	  if (!current->next())
		invalid = true;

	  return *this;
	}
	PartitionIterator& operator--() {
	  assert(current != nullptr);
	  assert(!current->isMin());
	  if (invalid)
		current->setMax();
	  else
		current->prev();
	  return *this;	
	}

	PartitionIterator operator++(int) {
	  PartitionIterator response(*this);
	  this->operator++();
	  return response;
	}
	PartitionIterator operator--(int) {
	  PartitionIterator response(*this);
	  this->operator--;
	  return response;
	}

	bool operator==(const PartitionIterator& rhs) const {
	  return (invalid && rhs.invalid) || (!(invalid || rhs.invalid) && *current == *(rhs.current));
	}
	bool operator!=(const PartitionIterator& rhs) const {
	  return (invalid != rhs.invalid) || (!(invalid || rhs.invalid) && *current != *(rhs.current));
	}
	bool operator<(const PartitionIterator& rhs) const {
	  if (invalid) {
		return false;
	  }
	  if (rhs.invalid) {
		return true;
	  }
	  return *current < *(rhs.current);
	}
	bool operator<=(const PartitionIterator& rhs) const {
	  if (invalid) {
		return rhs.invalid;
	  }
	  if (rhs.invalid) {
		return true;
	  }
	  return *current <= *(rhs.current);
	}
	bool operator>(const PartitionIterator& rhs) const {
	  return !(*this <= rhs);
	}
	bool operator>=(const PartitionIterator& rhs) const {
	  return !(*this < rhs);
	}

	const Partition<uint>& operator*() const {
	  return *current;
	}
	const Partition<uint>& operator->() const {
	  return *current;
	}

};


