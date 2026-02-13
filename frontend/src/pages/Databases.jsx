import { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { Upload, FileText, Trash2, Loader2 } from 'lucide-react';

const Databases = () => {
    const { user } = useAuth();
    const [databases, setDatabases] = useState([]);
    const [loading, setLoading] = useState(true);
    const [uploading, setUploading] = useState(false);
    const [file, setFile] = useState(null);
    const [name, setName] = useState('');

    useEffect(() => {
        fetchDatabases();
    }, [user]);

    const fetchDatabases = async () => {
        if (!user) return;
        try {
            const res = await axios.get(`http://localhost:8000/user/${user.id}/databases`);
            setDatabases(res.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleUpload = async (e) => {
        e.preventDefault();
        if (!file || !name) return;

        setUploading(true);
        const formData = new FormData();
        formData.append('name', name);
        formData.append('user_id', user.id);
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:8000/database', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            console.log('Upload successful:', response.data);
            setName('');
            setFile(null);
            fetchDatabases();
        } catch (err) {
            console.error('Upload error:', err);
            const errorMessage = err.response?.data?.detail || err.message || 'Upload failed';
            alert(`Upload failed: ${errorMessage}`);
        } finally {
            setUploading(false);
        }
    };

    const handleDelete = async (id) => {
        if (!confirm("Are you sure?")) return;
        try {
            await axios.delete(`http://localhost:8000/database/${id}`);
            fetchDatabases();
        } catch (err) {
            console.error(err);
        }
    }

    return (
        <div className="space-y-8">
            <div className="flex justify-between items-center">
                <h2 className="text-3xl font-bold tracking-tight">Databases</h2>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {/* Upload Card */}
                <div className="p-6 bg-card rounded-lg border border-border shadow-sm">
                    <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                        <Upload size={20} />
                        <span>Upload New Data</span>
                    </h3>
                    <form onSubmit={handleUpload} className="space-y-4">
                        <div>
                            <input
                                type="text"
                                placeholder="Dataset Name"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                className="w-full px-3 py-2 bg-input border border-border rounded-md text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                                required
                            />
                        </div>
                        <div className="border-2 border-dashed border-border rounded-md p-4 text-center cursor-pointer hover:border-primary transition-colors relative">
                            <input
                                type="file"
                                accept=".csv"
                                onChange={(e) => setFile(e.target.files[0])}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                            />
                            <p className="text-sm text-muted-foreground">{file ? file.name : "Click to select CSV"}</p>
                        </div>
                        <button
                            type="submit"
                            disabled={uploading}
                            className="w-full py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors disabled:opacity-50"
                        >
                            {uploading ? <Loader2 className="animate-spin mx-auto" /> : 'Upload'}
                        </button>
                    </form>
                </div>

                {/* Database List */}
                {loading ? (
                    <div className="col-span-full text-center p-8 text-muted-foreground">Loading...</div>
                ) : databases.map((db) => (
                    <div key={db.id} className="p-6 bg-card rounded-lg border border-border shadow-sm flex flex-col justify-between">
                        <div>
                            <h3 className="text-xl font-bold mb-2 flex items-center space-x-2">
                                <FileText size={20} className="text-primary" />
                                <span>{db.name}</span>
                            </h3>
                            <p className="text-xs text-muted-foreground break-all">{db.file_path}</p>
                        </div>
                        <div className="mt-4 flex justify-end">
                            <button
                                onClick={() => handleDelete(db.id)}
                                className="p-2 text-muted-foreground hover:text-destructive transition-colors"
                            >
                                <Trash2 size={16} />
                            </button>
                        </div>
                    </div>
                ))}
                {!loading && databases.length === 0 && (
                    <div className="col-span-full text-center p-8 text-muted-foreground border border-dashed border-border rounded-lg">
                        No databases found. Upload one to get started.
                    </div>
                )}
            </div>
        </div>
    );
};

export default Databases;
