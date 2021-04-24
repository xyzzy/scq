#include <stddef.h>
#include <assert.h>
#include <malloc.h>

//typedef struct octNode_t octNode_t, *oct_node;

/**
 * Node of an octree
 */
typedef struct octNode {
	// colour
	int r, g, b;

	// number of pixels using this colour
	int count;

	// link to parent and kids
	struct octNode *parent, *kids[8];

	// offset of this node within heap
	int ofsHeap;

	// number of active kids
	unsigned char numKids;

	// index of (this) kid within parent
	unsigned char idxKid;

//	int heap_idx;
//	unsigned char n_kids, kid_idx, inheap, depth;
} octNode_t;

/**
 * Heap for mapping tree to track nodes with lowest count
 */
typedef struct {
	// maximum capacity
	int size;

	// number of nodes in heap
	int count;

	// pointer to list
	octNode_t **ppNodes;
} octHeap_t;

octHeap_t  nodeHeap = {0, 0, NULL };
octNode_t *nodePool = NULL;
octNode_t *nodeRoot = NULL;

/**
 * Allocate a new kid for a tree node
 *
 * @return
 */
octNode_t* nodeNew(void) {
	static int left = 0;

	if (left <= 1) {
		// allocate a block of nodes
		left = 2048;
		octNode_t *block = (octNode_t*) calloc(sizeof(octNode_t), left);

		// reserve first entry to link blocks
		if (nodePool)
			nodePool->parent = block;
		nodePool = block;
	}

	// grab a kid
	octNode_t *kid = nodePool + --left;

	return kid;
}

/**
 * Insert RGB combo into tree
 *
 * @param {int} R
 * @param {int} G
 * @param {int} B
 * @return octNode_t *
 */
octNode_t *nodeInsert(int R, int G, int B) {

	// create tree root for first time
	if (nodeRoot == NULL)
		nodeRoot = nodeNew();

	octNode_t *pNode = nodeRoot;

	// walk through tree
	for (int bit = 7; bit >= 0; --bit) {
		// slice RGB to kid index
		int ix = ((R >> (bit - 2)) & 4) | ((G >> (bit - 1)) & 2) | ((B >> (bit - 0)) & 1);

		// walk futher down the tree
		if (!pNode->kids[ix]) {
			//  create new intermediate
			octNode_t *pKid = nodeNew();
			pKid->idxKid = ix;
			pKid->parent = pNode;
			// add to parent
			pNode->numKids++;
			pNode->kids[ix] = pKid;
		}

		pNode = pNode->kids[ix];
	}

	// update the fields
	pNode->r += R;
	pNode->g += G;
	pNode->b += B;
	pNode->count++;

	// return node
	return pNode;
}

int nodeCompare(octNode_t* pLeft, octNode_t* pRight) {
	if (pLeft->numKids > pRight->numKids)
		return +1;
	if (pLeft->numKids < pRight->numKids)
		return -1;
	return pLeft->count - pRight->count;
}
/**
 * Order the path between the node and start of the heap
 *
 * @param pHeap
 * @param pNode
 */
void heapUp(octHeap_t *pHeap, octNode_t* pNode) {
	// get offset of node in heap
	int ofsHeap = pNode->ofsHeap;

	// walk to start of heap and find insert point
	while (ofsHeap > 1) {
		// find previous insert point
		octNode_t *pPrev = pHeap->ppNodes[ofsHeap / 2];
		if (nodeCompare(pPrev, pNode) <= 0)
			break;

		// relocate contents of next iteration
		pHeap->ppNodes[ofsHeap] = pPrev;
		pPrev->ofsHeap = ofsHeap;
		ofsHeap /= 2;
	}

	pNode->ofsHeap = ofsHeap;
	pHeap->ppNodes[ofsHeap] = pNode;
}

/**
 * Order the path between the node and end of the heap
 *
 * @param pHeap
 * @param pNode
 */
void heapDown(octHeap_t *pHeap, octNode_t *pNode) {
	// get offset of node in pHeap
	int ofsHeap = pNode->ofsHeap;

	// walk to end of heap and find insert point
	while (ofsHeap * 2 < pHeap->count) {
		// find next insert point
		int next = ofsHeap * 2;

		// select odd/even, which ever is lower
		if (next + 1 < pHeap->count && nodeCompare(pHeap->ppNodes[next], pHeap->ppNodes[next + 1]) > 0)
			next++;

		if (nodeCompare(pNode, pHeap->ppNodes[next]) <= 0)
			break;

		// relocate contents of next iteration
		pHeap->ppNodes[ofsHeap] = pHeap->ppNodes[next];
		pHeap->ppNodes[ofsHeap]->ofsHeap = ofsHeap;
		ofsHeap = next;
	}

	// Populate insert point
	pHeap->ppNodes[ofsHeap] = pNode;
	pNode->ofsHeap = ofsHeap;
}

void heapAdd(octHeap_t *pHeap, octNode_t *pNode) {
	if (pNode->ofsHeap > 0) {
		// update position
		heapDown(pHeap, pNode);
		heapUp(pHeap, pNode);
		return;
	}

	// first entry in heap is reserved
	if (!pHeap->count)
		pHeap->count = 1;

	// expand heap size
	if (pHeap->count >= pHeap->size) {
		while (pHeap->count >= pHeap->size)
			pHeap->size += 256000;
		pHeap->ppNodes = (octNode_t **) realloc(pHeap->ppNodes, sizeof(*pHeap->ppNodes) * pHeap->size);
	}

	pNode->ofsHeap = pHeap->count;
	pHeap->ppNodes[pHeap->count++] = pNode;

	// update position
	heapUp(pHeap, pNode);
}

octNode_t * heapPop(octHeap_t *pHeap) {
	// test if heap contains something
	if (pHeap->count <= 1)
		return 0;

	// find endpoint node with lowest count
	int ix;
	for (ix=1; ix<pHeap->count; ix++) {
		if (pHeap->ppNodes[ix]->numKids == 0)
			break;
	}

	// get the head with lowest count
	octNode_t *pRet = pHeap->ppNodes[ix];
	// fill head with popped tail
	pHeap->ppNodes[ix] = pHeap->ppNodes[--pHeap->count];
	pHeap->ppNodes[ix]->ofsHeap = 1;
	// clear tail
	pHeap->ppNodes[pHeap->count] = NULL;

	// re-order by scanning to tail
	heapDown(pHeap, pHeap->ppNodes[ix]);

	return pRet;
}

/**
 * Fold leaf node into parent
 *
 * @param pNode
 * @return
 */
octNode_t* nodeFold(octNode_t* pNode) {
	assert(pNode->numKids == 0);

	// get pointer to parent
	octNode_t*pParent = pNode->parent;

	// merge kid into parent
	pParent->count += pNode->count;
	pParent->r += pNode->r;
	pParent->g += pNode->g;
	pParent->b += pNode->b;

	// remove kid from parent
	pParent->numKids--;
	pParent->kids[pNode->idxKid] = 0;

	// if parent has no kids, then parent becomes an endpoint colour
	if (pParent->numKids == 0)
		heapAdd(&nodeHeap, pParent);

	return pParent;
}
