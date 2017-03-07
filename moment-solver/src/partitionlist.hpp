#include <cassert>      // assert
#include <iterator>     // iterator
#include <utility>      // swap
#include <Eigen/Dense>

#include "partition.hpp"

template <class uint>
class PartitionList {
	private:
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
					if (invalid) {
						current->setMax();
						invalid = false;
					}
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


		PartitionIterator *m_first, *m_last;

		Eigen::Array<uint, Eigen::Dynamic, Eigen::Dynamic> sizes;


		// helper function for finding the partition offset
		uint getIndex(const Partition<uint>& n) const {
			uint index = 0;

			uint k = n.getLength();
			uint m = n.getMass();

			for (uint i = k; i > 2; i--) {
				for (uint j = 0; j < n.get(i); j++) {
					index += sizes(m - i*j, i-3);
				}
				m -= i * n.get(i);
			}
			index += n.get(2);
			return index;
		}
	public:
		typedef PartitionIterator iterator;
		typedef PartitionIterator const_iterator;


		PartitionList() : m_first(nullptr), m_last(nullptr) {}
		PartitionList(size_t len, uint mass) {
			Partition<uint> a(len, mass);
		  	m_first = new PartitionIterator(a);
			a.setMax();
			m_last = new PartitionIterator(a);
			++(*m_last);


			uint num = (uint) (len*(len+1)/2+1);
			Eigen::Array<uint, Eigen::Dynamic, 1> c;
			c = decltype(c)::Zero(num);
			sizes = decltype(sizes)::Zero(mass+1, len-1);
			sizes.row(0).setConstant(1);
			sizes.col(0) = decltype(c)::LinSpaced(mass+1,0,mass) / 2 + 1;
			c(0) = 1; c(1) = -1; c(2) = -1; c(3) = 1;

			for (size_t i = 3; i <= len; i++) {
				uint cnum = (uint) (i*(i+1)/2);
				c.segment(i, cnum+1-i) -= c.head(cnum+1-i).eval();

				if (cnum > mass) {
					for (uint j = 1; j <= mass; j++)
						sizes(j, i-2) = -c.segment(1, j).reverse().matrix().dot(sizes.col(i-2).head(j).matrix());
				}
				else {
					for (uint j = 1; j < cnum; j++)
						sizes(j, i-2) = -c.segment(1, j).reverse().matrix().dot(sizes.col(i-2).head(j).matrix());
					for (uint j = cnum; j <= mass; j++)
						sizes(j, i-2) = -c.segment(1, cnum).reverse().matrix().dot(sizes.col(i-2).segment(j-cnum,cnum).matrix());
				}
			}


		}
		~PartitionList() {
			if (m_first != nullptr)
				delete m_first;
			if (m_last != nullptr)
				delete m_last;
		}

		uint indexOf(const Partition<uint>& n) const {
			uint index = 0;
			uint m = n.getMass();
			for (uint i = n.getLength(); i > 2; i--) {
				for (uint j = 0; j < n.get(i); j++) {
					index += sizes(m - i*j, i-3);
				}
				m -= i * n.get(i);
			}
			index += n.get(2);
			return index;
		}
		uint size() const {
			uint mass = (**m_first).getMass();
			uint len = (**m_first).getLength();
			return sizes(mass, len-2);
		}

		const_iterator begin() const {
			return iterator(*m_first);
		}
		const_iterator end() const {
			return iterator(*m_last);
		}
		const_iterator cbegin() const {
			return const_iterator(*m_first);
		}
		const_iterator cend() const {
			return const_iterator(*m_last);
		}
};

