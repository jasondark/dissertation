#include <cassert>
#include <ostream>

// A partition is a non-negative vector satisfying Sum[i*n[i-1],{i,0,len-1}] == mass
// There is a natural ordering, and we supply methods to mutate it with ease.
template <class uint>
struct Partition {
	size_t len;
	uint mass;
	uint* vals;

	bool increment(uint i) {
		if (vals[0] >= i) {
			vals[0] -= i;
			vals[i-1]++;
			return true;
		}
		return false;
	}
	bool decrement(uint i) {
		if (vals[i-1] > 0) {
			vals[i-1]--;
			vals[0] += i;
			return true;
		}
		return false;
	}
	bool set(size_t i, uint k) {
		if (k == vals[i-1]) {
			return true;
		}
		if (k < vals[i-1]) {
			vals[0] += (vals[i-1]-k) * (uint) i;
			vals[i-1] = k;
			return true;
		}
		if (k > vals[i-1]) {
			uint delta = (k-vals[i-1]) * (uint) i;
			if (vals[0] < delta)
				return false;
			vals[0] -= delta;
			vals[i-1] = k;
			return true;
		}

		return false;
	}

	Partition() : len(size_t(0)), mass(uint(0)), vals(nullptr) {}
	Partition(size_t len_, uint mass_) : len(len_), mass(mass_) {
		vals = new uint[len];
		vals[0] = mass;
		for (size_t i = 1; i < len; i++)
			vals[i] = uint(0);
	}
	Partition(const Partition& rhs) : len(rhs.len), mass(rhs.mass) {
		vals = new uint[len];
		for (size_t i = 0; i < rhs.len; i++)
			vals[i] = rhs.vals[i];
	}
	~Partition() {
		if (vals != nullptr)
			delete[] vals;
	}


	uint get(size_t i) const {
		assert(i <= len);
		assert(i > size_t(0));
		return vals[i-1];
	}

	uint getLength() const {
		return len;
	}
	uint getMass() const {
		return mass;
	}


	bool isMin() const {
		return get(1) == mass;
	}
	bool isMax() const {
		uint m = mass;
		for (size_t i = len; i > 1; i--) {
			if (get(i) != m / i)
				return false;
			m -= i*get(i);
		}
		return true;
	}
	void setMin() {
		for (size_t i = 2; i <= len; i++)
			set(i,0);
	}
	void setMax() {
		setMin();
		for (size_t i = len; i > 1; i--)
			set(i, get(1)/i);
	}
	bool next() {
		if (isMax())
			return false;

		uint k = (uint) len;
		uint i = 2;
		while (get(1) < i) {
			if (i < k) {
				set(i, 0);
				i++;
			}
		}
		increment(i);
		return true;
	}
	bool prev() {
		if (isMin())
			return false;

		size_t j, k;
		for (k = 2; k <= len; k++) {
			if (decrement(k)) {
				for (j = k-1; j > 1; j--) {
					set(j, get(1)/j);
				}
				return true;
			}
		}
		return false;
	}



	Partition& operator=(const Partition& rhs) {
		if (len != rhs.len) {
			if (vals != nullptr)
				delete[] vals;
			len = rhs.len;
			vals = new uint[len];
		}
		mass = rhs.mass;
		for (size_t i = 0; i < rhs.len; i++)
			vals[i] = rhs.vals[i];

		return *this;
	}

	bool operator==(const Partition& rhs) const {
		assert(rhs.len == len && rhs.mass == mass);

		for (size_t i = 0; i < len; i++) {
			if (vals[i] != rhs.vals[i])
				return false;
		}

		return true;
	}
	bool operator!=(const Partition& rhs) const {
		return !(*this == rhs);
	}

	bool operator<(const Partition& rhs) const {
		assert(rhs.len == len && rhs.mass == mass);

		for (size_t i = len-1; i >= 0; i--) {
			if (vals[i] < rhs.vals[i])
				return true;
			else if (vals[i] > rhs.vals[i])
				return false;
		}
		return false;
	}
	bool operator<=(const Partition& rhs) const {
		assert(rhs.len == len && rhs.mass == mass);

		for (size_t i = len-1; i >= 0; i--) {
			if (vals[i] < rhs.vals[i])
				return true;
			else if (vals[i] > rhs.vals[i])
				return false;
		}
		return true;
	}
	bool operator>(const Partition& rhs) const {
		return !(*this <= rhs);
	}
	bool operator>=(const Partition& rhs) const {
		return !(*this < rhs);
	}

	friend std::ostream& operator<<(std::ostream& os, const Partition<uint>& part) {
		if (part.len == size_t(0)) {
			os << "()";
		}
		else {
			os << '(' << part.vals[0];
			for (size_t i = 1; i < part.len; i++)
				os << ',' << part.vals[i];
			os << ')';
		}
		return os;
	}
};


