import { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { Link } from 'react-router-dom';
import { Plus, Play, Loader2, FlaskConical } from 'lucide-react';

const Experiments = () => {
    const { user } = useAuth();
    const [experiments, setExperiments] = useState([]);
    const [databases, setDatabases] = useState([]);
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);
    const [columns, setColumns] = useState([]);
    const [loadingColumns, setLoadingColumns] = useState(false);

    // Form State
    const [name, setName] = useState('');
    const [selectedDb, setSelectedDb] = useState('');
    const [target, setTarget] = useState('');
    const [features, setFeatures] = useState('all'); // Default to all for now
    const [models, setModels] = useState('all');

    useEffect(() => {
        if (user) {
            fetchDatabases();
            fetchExperiments();
        }
    }, [user]);

    const fetchDatabases = async () => {
        try {
            const res = await axios.get(`http://localhost:8000/user/${user.id}/databases`);
            setDatabases(res.data);
            if (res.data.length > 0) {
                setSelectedDb(res.data[0].id);
                loadColumnsForDatabase(res.data[0].id);
            }
        } catch (err) {
            console.error(err);
        }
    };

    const loadColumnsForDatabase = async (dbId) => {
        try {
            setLoadingColumns(true);
            const db = databases.find(d => d.id === dbId);
            if (!db) return;
            
            // Fetch the CSV file and parse columns from header
            const response = await fetch(db.file_path);
            const csv = await response.text();
            const headerLine = csv.split('\n')[0];
            const cols = headerLine.split(',').map(col => col.trim());
            setColumns(cols);
            setTarget(''); // Reset target when changing database
        } catch (err) {
            console.error('Error loading columns:', err);
            setColumns([]);
        } finally {
            setLoadingColumns(false);
        }
    };

    const fetchExperiments = async () => {
        // ideally we have an endpoint for user experiments, but we can iterate dbs
        // for fast protyping:
        if (databases.length === 0) return;

        let allExps = [];
        // This is inefficient but works for MVP without backend change
        // Better: GET /user/{id}/experiments
        // For now, let's just fetch for the current user's DBs if we can
        // Actually, let's just show experiments for the SELECTED DB or all?
        // Let's rely on the user selecting a DB to filter experiments.

        // Wait, I didn't add GET /user/experiments.
        // I added GET /database/{id}/experiments.
        // So I'll fetch for each DB.
    };

    // Better strategy: Just load experiments when a DB is selected or try to load all
    useEffect(() => {
        const loadAll = async () => {
            if (databases.length === 0) return;
            setLoading(true);
            try {
                const promises = databases.map(db => axios.get(`http://localhost:8000/database/${db.id}/experiments`));
                const results = await Promise.all(promises);
                const flat = results.flatMap(r => r.data);
                setExperiments(flat);
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        }
        loadAll();
    }, [databases]);


    const handleCreate = async (e) => {
        e.preventDefault();
        setCreating(true);
        try {
            const payload = {
                name,
                database_id: selectedDb,
                target,
                features,
                models
            };
            await axios.post('http://localhost:8000/experiment', payload);
            setName('');
            setTarget('');
            // reload
            const promises = databases.map(db => axios.get(`http://localhost:8000/database/${db.id}/experiments`));
            const results = await Promise.all(promises);
            setExperiments(results.flatMap(r => r.data));
        } catch (err) {
            console.error(err);
            alert("Failed to create experiment");
        } finally {
            setCreating(false);
        }
    };

    const handleRun = async (id) => {
        try {
            await axios.post(`http://localhost:8000/experiment/${id}/run`);
            // trigger reload or status update
            const promises = databases.map(db => axios.get(`http://localhost:8000/database/${db.id}/experiments`));
            const results = await Promise.all(promises);
            setExperiments(results.flatMap(r => r.data));
        } catch (e) {
            console.error(e);
            alert("Failed to run");
        }
    };

    return (
        <div className="space-y-8">
            <h2 className="text-3xl font-bold tracking-tight">Experiments</h2>

            {/* Create Form */}
            <div className="p-6 bg-card rounded-lg border border-border">
                <h3 className="text-lg font-semibold mb-4">New Experiment</h3>
                <form onSubmit={handleCreate} className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 items-end">
                    <div className="space-y-2">
                        <label className="text-sm font-medium">Experiment Name</label>
                        <input
                            className="w-full px-3 py-2 bg-input border border-border rounded-md"
                            value={name} onChange={e => setName(e.target.value)} required
                        />
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium">Database</label>
                        <select
                            className="w-full px-3 py-2 bg-input border border-border rounded-md"
                            value={selectedDb} 
                            onChange={e => {
                                setSelectedDb(e.target.value);
                                loadColumnsForDatabase(parseInt(e.target.value));
                            }} 
                            required
                        >
                            <option value="" disabled>Select DB</option>
                            {databases.map(db => <option key={db.id} value={db.id}>{db.name}</option>)}
                        </select>
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium">Target Column</label>
                        {loadingColumns ? (
                            <div className="w-full px-3 py-2 bg-input border border-border rounded-md text-muted-foreground flex items-center">
                                <Loader2 className="animate-spin mr-2" size={16} />
                                Loading columns...
                            </div>
                        ) : (
                            <select
                                className="w-full px-3 py-2 bg-input border border-border rounded-md"
                                value={target} 
                                onChange={e => setTarget(e.target.value)} 
                                required
                            >
                                <option value="" disabled>Select target column</option>
                                {columns.map(col => <option key={col} value={col}>{col}</option>)}
                            </select>
                        )}
                    </div>
                    <button
                        type="submit"
                        disabled={creating}
                        className="flex items-center justify-center space-x-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
                    >
                        {creating ? <Loader2 className="animate-spin" /> : <Plus size={18} />}
                        <span>Create</span>
                    </button>
                </form>
            </div>

            {/* List */}
            <div className="grid gap-4">
                {experiments.map(exp => (
                    <div key={exp.id} className="p-4 bg-card rounded-lg border border-border flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                            <div className="p-2 bg-muted rounded-full">
                                <FlaskConical size={24} />
                            </div>
                            <div>
                                <h4 className="font-bold">{exp.name}</h4>
                                <div className="text-sm text-muted-foreground flex space-x-4">
                                    <span>Target: {exp.target}</span>
                                    <span className={`capitalize px-2 py-0.5 rounded-full text-xs border ${exp.status === 'completed' ? 'bg-green-900/20 text-green-400 border-green-900' :
                                            exp.status === 'running' ? 'bg-blue-900/20 text-blue-400 border-blue-900' :
                                                exp.status === 'failed' ? 'bg-red-900/20 text-red-400 border-red-900' :
                                                    'bg-gray-800 text-gray-400 border-gray-700'
                                        }`}>
                                        {exp.status}
                                    </span>
                                </div>
                            </div>
                        </div>

                        <div className="flex items-center space-x-2">
                            {exp.status === 'pending' && (
                                <button
                                    onClick={() => handleRun(exp.id)}
                                    className="p-2 bg-secondary hover:bg-secondary/80 rounded-md text-secondary-foreground"
                                    title="Run Experiment"
                                >
                                    <Play size={18} />
                                </button>
                            )}
                            <Link
                                to={`/experiments/${exp.id}`}
                                className="px-4 py-2 text-sm font-medium text-primary hover:text-primary/80"
                            >
                                View Details
                            </Link>
                        </div>
                    </div>
                ))}
                {!loading && experiments.length === 0 && (
                    <div className="text-center p-8 text-muted-foreground border border-dashed border-border rounded-lg">
                        No experiments found.
                    </div>
                )}
            </div>
        </div>
    );
};

export default Experiments;
