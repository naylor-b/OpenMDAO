// <<hpp_insert gen/ModelData.js>>
// <<hpp_insert src/OmTreeNode.js>>

/** Process the tree, connections, and other info provided about the model. */
class OmModelData extends ModelData {
    /** Do some discovery in the tree and rearrange & enhance where necessary. */
    constructor(modelJSON) {
        super(modelJSON);

        if (this.unconnectedInputs > 0)
            console.info("Unconnected nodes: ", this.unconnectedInputs);

        this._initSubSystemChildren(this.root);
        this._updateAutoIvcNames();

        debugInfo("New model: ", this);
        // this.errorCheck();
    }

    /** Tasks to perform early from the superclass constructor */
    _init(modelJSON) {
        modelJSON.tree.name = 'model'; // Change 'root' to 'model'
        this.abs2prom = modelJSON.abs2prom; // May be undefined.
        this.declarePartialsList = modelJSON.declare_partials_list;
        this.useDeclarePartialsList = (this.declarePartialsList.length > 0);
        this.sysPathnamesList = modelJSON.sys_pathnames_list;

        this.unconnectedInputs = 0;
        this.autoivcSources = 0;
        this.md5_hash = modelJSON.md5_hash; // compute here instead of python?
    }

    /**
     * For debugging: Make sure every tree member is an N2TreeNode.
     * @param {N2TreeNode} [node = this.root] The node to start with.
     */
    errorCheck(node = this.root) {
        if (!(node instanceof N2TreeNode))
            debugInfo('Node with problem: ', node);

        for (const prop of ['parent', 'originalParent', 'parentComponent']) {
            if (node[prop] && !(node[prop] instanceof N2TreeNode))
                debugInfo('Node with problem ' + prop + ': ', node);
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                this.errorCheck(child);
            }
        }
    }

    _newNode(element, attribNames) {
        return new OmTreeNode(element, attribNames);
    }

    /**
     * Sets parents and depth of all nodes, and determine max depth. Flags the
     * parent node as implicit if the node itself is implicit.
     * @param {N2TreeNode} node Item to process.
     * @param {N2TreeNode} parent Parent of node, null for root node.
     * @param {number} depth Numerical level of ancestry.
     */
     _setParentsAndDepth(node, parent, depth) {
        super._setParentsAndDepth(node, parent, depth);

        if (this.abs2prom.input[node.uuid] !== undefined) {
            node.promotedName = this.abs2prom.input[node.uuid];
        }
        else if (this.abs2prom.output[node.uuid] !== undefined) {
            node.promotedName = this.abs2prom.output[node.uuid];
        }

        this.identifyUnconnectedInput(node);
        if (node.isInputOrOutput()) {
            const parentComponent = (node.originalParent) ? node.originalParent : node.parent;
            if (parentComponent.type == "subsystem" &&
                parentComponent.subsystem_type == "component") {
                node.parentComponent = parentComponent;
            }
            else {
                throw ("Input or output without a parent component!");
            }
        }

        if (node.isSubsystem()) {
            this.maxSystemDepth = Math.max(depth, this.maxSystemDepth);
        }

        if (parent && node.implicit) { parent.implicit = true; }
    }

    hasAutoIvcSrc(elementPath) {
        for (const conn of this.conns) {
            if (conn.tgt == elementPath && conn.src.match(/^_auto_ivc.*$/)) {
                debugInfo(elementPath + " source is an auto-ivc output.");
                this.autoivcSources++;
                return true;
            }
        }

        return false;
    }

    /**
     * Find the target of an Auto-IVC variable.
     * @param {String} elementPath The full path of the element to check. Must start with _auto_ivc.
     * @return {String} The absolute path of the target element, or undefined if not found.
     */
    getAutoIvcTgt(elementPath) {
        if (!elementPath.match(/^_auto_ivc.*$/)) return undefined;

        for (const conn of this.conns) {
            if (conn.src == elementPath) {
                return conn.tgt;
            }
        }

        console.warn(`No target connection found for ${elementPath}.`)
        return undefined;
    }

    /**
     * Create an array in each node containing references to its
     * children that are subsystems. Runs recursively over the node's
     * children array.
     * @param {N2TreeNode} node Node with children to check.
     */
    _initSubSystemChildren(node) {
        if (!node.hasChildren()) {
            return;
        }

        for (const child of node.children) {
            if (child.isSubsystem()) {
                if (!node.hasChildren('subsystem_children'))
                    node.subsystem_children = [];

                node.subsystem_children.push(child);
                this._initSubSystemChildren(child);
            }
        }
    }

    /**
     * Build a string from the absoluate path names of the two elements and
     * try to find it in the declare partials list.
     * @param {Object} srcObj The source element.
     * @param {Object} tgtObj The target element.
     * @return {Boolean} True if the string was found.
     */
    isDeclaredPartial(srcObj, tgtObj) {
        let partialsStr = tgtObj.absPathName + " > " + srcObj.absPathName;

        return this.declarePartialsList.includes(partialsStr);
    }

    /**
     * The cycle_arrows object in each connection is an array of length-2 arrays,
     * each of which is an index into the sysPathnames array. Using that array we
     * can resolve the indexes to pathnames to the associated objects.
     * @param {Object} conn Reference to the connection to operate on.
     */
    _additionalConnProcessing(conn, srcObj, tgtObj) {
        const sysPathnames = this.sysPathnamesList;
        const throwLbl = 'ModelData._computeConnections: ';

        if (Array.isPopulatedArray(conn.cycle_arrows)) {
            const cycleArrowsArray = [];
            for (const cycleArrow of conn.cycle_arrows) {
                if (cycleArrow.length != 2) {
                    console.warn(throwLbl + "cycleArrowsSplitArray length not 2, got " +
                        cycleArrow.length + ": " + cycleArrow);
                    continue;
                }

                const srcPathname = sysPathnames[cycleArrow[0]];
                const tgtPathname = sysPathnames[cycleArrow[1]];

                const arrowBeginObj = this.nodePaths[srcPathname];
                if (!arrowBeginObj) {
                    console.warn(throwLbl + "Cannot find cycle arrows begin object " + srcPathname);
                    continue;
                }

                const arrowEndObj = this.nodePaths[tgtPathname];
                if (!arrowEndObj) {
                    console.warn(throwLbl + "Cannot find cycle arrows end object " + tgtPathname);
                    continue;
                }

                cycleArrowsArray.push({
                    "begin": arrowBeginObj,
                    "end": arrowEndObj
                });
            }

            if (!tgtObj.parent.hasOwnProperty("cycleArrows")) {
                tgtObj.parent.cycleArrows = [];
            }
            tgtObj.parent.cycleArrows.push({
                "src": srcObj,
                "arrows": cycleArrowsArray
            });
        }
    }

    /**
     * If the Auto-IVC component exists, rename its child variables to their
     * promoted names so they can be easily recognized instead of as v0, v1, etc.
     */
    _updateAutoIvcNames() {
        const aivc = this.nodePaths['_auto_ivc'];
        if (aivc !== undefined && aivc.hasChildren()) {
            for (const ivc of aivc.children) {
                if (!ivc.isFilter()) {
                    const tgtPath = this.getAutoIvcTgt(ivc.absPathName);

                    if (tgtPath !== undefined) {
                        ivc.promotedName = this.nodePaths[tgtPath].promotedName;
                    }
                }
            }
        }
    }

    /**
     * If an element has no connection naming it as a source or target,
     * relabel it as unconnected.
     * @param {N2TreeNode} node The tree node to work on.
     */
    identifyUnconnectedInput(node) {
        if (!node.hasOwnProperty('uuid')) {
            console.warn("identifyUnconnectedInput error: uuid not set for ", node);
        }
        else {
            if (node.isInput()) {
                if (!node.hasChildren() && !this.hasAnyConnection(node.uuid))
                    node.type = "unconnected_input";
                else if (this.hasAutoIvcSrc(node.uuid))
                    node.type = "autoivc_input";
            }
        }
    }
}
