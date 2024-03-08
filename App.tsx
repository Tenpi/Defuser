import React, {useEffect, useState, useContext, useReducer} from "react"
import {Switch, Route, Redirect, useHistory, useLocation} from "react-router-dom"
import Context, {EnableDragContext, MobileContext, SocketContext, ImagesContext, UpdateImagesContext, TextualInversionsContext,
HypernetworksContext, LorasContext, ReverseSortContext, NSFWImagesContext, ImageInputImagesContext, NovelAIImagesContext,
NovelAINSFWImagesContext, NovelAIImageInputImagesContext, HolaraAIImagesContext, HolaraAINSFWImagesContext, HolaraAIImageInputImagesContext, 
GeneratorContext, SavedPromptsContext, SavedPromptsNovelAIContext, SavedPromptsHolaraAIContext, ModelDirContext, OutputDirContext} from "./Context"
import axios from "axios"
import {io} from "socket.io-client"
import functions from "./structures/Functions"
import MainPage from "./pages/MainPage"
import "./index.less"

require.context("./assets/icons", true)

const App: React.FunctionComponent = (props) => {
    const [ignored, forceUpdate] = useReducer(x => x + 1, 0)
    const [loaded, setLoaded] = useState(false)
    const [enableDrag, setEnableDrag] = useState(false)
    const [mobile, setMobile] = useState(false)
    const [socket, setSocket] = useState(null) as any
    const [images, setImages] = useState([])
    const [nsfwImages, setNSFWImages] = useState([])
    const [imageInputImages, setImageInputImages] = useState([])
    const [novelAIImages, setNovelAIImages] = useState([])
    const [novelAINSFWImages, setNovelAINSFWImages] = useState([])
    const [novelAIImageInputImages, setNovelAIImageInputImages] = useState([])
    const [holaraAIImages, setHolaraAIImages] = useState([])
    const [holaraAINSFWImages, setHolaraAINSFWImages] = useState([])
    const [holaraAIImageInputImages, setHolaraAIImageInputImages] = useState([])
    const [updateImages, setUpdateImages] = useState(true)
    const [textualInversions, setTextualInversions] = useState([])
    const [hypernetworks, setHypernetworks] = useState([])
    const [loras, setLoras] = useState([])
    const [reverseSort, setReverseSort] = useState(false)
    const [generator, setGenerator] = useState("local")
    const [savedPrompts, setSavedPrompts] = useState([])
    const [savedPromptsNovelAI, setSavedPromptsNovelAI] = useState([])
    const [savedPromptsHolaraAI, setSavedPromptsHolaraAI] = useState([])
    const [modelDir, setModelDir] = useState("models")
    const [outputDir, setOutputDir] = useState("outputs")

    useEffect(() => {
        functions.preventDragging()
        setTimeout(() => {
            setLoaded(true)
        }, 100)
    }, [])

    const history = useHistory()
    const location = useLocation()

    useEffect(() => {
        setTimeout(() => {
            if (mobile) {
                if (enableDrag) functions.dragScroll(false)
                return
            }
            functions.dragScroll(enableDrag)
        }, 100)
    }, [enableDrag, mobile, history])

    useEffect(() => {
        const mobileQuery = (query: any) => {
            if (query.matches) {
                setMobile(true)
            } else {
                setMobile(false)
            }
        }
        const media = window.matchMedia("(max-width: 700px)")
        media.addEventListener("change", mobileQuery)
        mobileQuery(media)
        document.documentElement.style.visibility = "visible"
    }, [])

    useEffect(() => {
        const socket = io()
        socket.on("connect", () => {
            setSocket(socket)
        })
        return () => {
            socket.disconnect()
        }
    }, [])

    useEffect(() => {
        if (!socket) return
        setTimeout(() => {
            let interrogatorName = localStorage.getItem("interrogatorName")
            if (!interrogatorName) interrogatorName = "wdtagger"
            const modelName = localStorage.getItem("modelName")
            const vaeName = localStorage.getItem("vaeName")
            const clipSkip = localStorage.getItem("clipSkip")
            const processing = localStorage.getItem("processing")
            const generator = localStorage.getItem("generator")
            socket.emit("load interrogate model", interrogatorName)
            socket.emit("load diffusion model", modelName, vaeName, clipSkip, processing, generator)
            socket.emit("load control models")
        }, 200)
    }, [socket])

    const initSaveData = async () => {
        let savedLocal = await axios.get("/saved-local-images").then((r) => r.data)
        localStorage.setItem("saved", JSON.stringify(savedLocal))
        let savedNovelAI = await axios.get("/saved-novelai-images").then((r) => r.data)
        localStorage.setItem("saved-novel-ai", JSON.stringify(savedNovelAI))
        let savedHolaraAI = await axios.get("/saved-holara-images").then((r) => r.data)
        localStorage.setItem("saved-holara-ai", JSON.stringify(savedHolaraAI))
        let savedLocalPrompts = await axios.get("/saved-local-prompts").then((r) => r.data)
        let savedNovelAIPrompts = await axios.get("/saved-novelai-prompts").then((r) => r.data)
        let savedHolaraAIPrompts = await axios.get("/saved-holara-prompts").then((r) => r.data)
        localStorage.setItem("savedPrompts", JSON.stringify(savedLocalPrompts))
        localStorage.setItem("savedPrompts-novel-ai", JSON.stringify(savedNovelAIPrompts))
        localStorage.setItem("savedPrompts-holara-ai", JSON.stringify(savedHolaraAIPrompts))
        setSavedPrompts(savedLocalPrompts)
        setSavedPromptsNovelAI(savedNovelAIPrompts)
        setSavedPromptsHolaraAI(savedHolaraAIPrompts)
    }

    useEffect(() => {
        initSaveData()
    }, [])

    const processImageUpdate = async () => {
        let images = await axios.get("/all-outputs").then((r) => r.data)
        setImages(images)
        let nsfwImages = await axios.get("/all-nsfw-outputs").then((r) => r.data)
        setNSFWImages(nsfwImages)
        let imageInputImages = await axios.get("/all-image-outputs").then((r) => r.data)
        setImageInputImages(imageInputImages)
        let novelAIImages = await axios.get("/all-novelai-outputs").then((r) => r.data)
        setNovelAIImages(novelAIImages)
        let novelAINSFWImages = await axios.get("/all-novelai-nsfw-outputs").then((r) => r.data)
        setNovelAINSFWImages(novelAINSFWImages)
        let novelAIImageInputImages = await axios.get("/all-novelai-image-outputs").then((r) => r.data)
        setNovelAIImageInputImages(novelAIImageInputImages)
        let holaraAIImages = await axios.get("/all-holara-outputs").then((r) => r.data)
        setHolaraAIImages(holaraAIImages)
        let holaraAINSFWImages = await axios.get("/all-holara-nsfw-outputs").then((r) => r.data)
        setHolaraAINSFWImages(holaraAINSFWImages)
        let holaraAIImageInputImages = await axios.get("/all-holara-image-outputs").then((r) => r.data)
        setHolaraAIImageInputImages(holaraAIImageInputImages)
    }

    useEffect(() => {
        if (updateImages) {
            processImageUpdate()
            setUpdateImages(false)
        }
    }, [updateImages])

    const processNetworkUpdate = async () => {
        const textualInversions = await axios.get("/textual-inversions").then((r) => r.data)
        setTextualInversions(textualInversions)
        const hypernetworks = await axios.get("/hypernetworks").then((r) => r.data)
        setHypernetworks(hypernetworks)
        const loras = await axios.get("/lora-models").then((r) => r.data)
        setLoras(loras)
    }

    useEffect(() => {
        setTimeout(() => {
            processNetworkUpdate()
        }, 200)
    }, [modelDir])

    useEffect(() => {
        axios.post("/update-model-dir", {model_dir: modelDir.trim()})
    }, [modelDir])

    useEffect(() => {
        axios.post("/update-output-dir", {output_dir: outputDir.trim()})
    }, [outputDir])

    return (
        <div className={`app ${!loaded ? "stop-transitions" : ""}`}>
            <OutputDirContext.Provider value={{outputDir, setOutputDir}}>
            <ModelDirContext.Provider value={{modelDir, setModelDir}}>
            <SavedPromptsContext.Provider value={{savedPrompts, setSavedPrompts}}>
            <SavedPromptsHolaraAIContext.Provider value={{savedPromptsHolaraAI, setSavedPromptsHolaraAI}}>
            <SavedPromptsNovelAIContext.Provider value={{savedPromptsNovelAI, setSavedPromptsNovelAI}}>
            <HolaraAIImageInputImagesContext.Provider value={{holaraAIImageInputImages, setHolaraAIImageInputImages}}>
            <HolaraAINSFWImagesContext.Provider value={{holaraAINSFWImages, setHolaraAINSFWImages}}>
            <HolaraAIImagesContext.Provider value={{holaraAIImages, setHolaraAIImages}}>
            <GeneratorContext.Provider value={{generator, setGenerator}}>
            <NovelAIImageInputImagesContext.Provider value={{novelAIImageInputImages, setNovelAIImageInputImages}}>
            <NovelAINSFWImagesContext.Provider value={{novelAINSFWImages, setNovelAINSFWImages}}>
            <NovelAIImagesContext.Provider value={{novelAIImages, setNovelAIImages}}>
            <ImageInputImagesContext.Provider value={{imageInputImages, setImageInputImages}}>
            <NSFWImagesContext.Provider value={{nsfwImages, setNSFWImages}}>
            <ReverseSortContext.Provider value={{reverseSort, setReverseSort}}>
            <LorasContext.Provider value={{loras, setLoras}}>
            <HypernetworksContext.Provider value={{hypernetworks, setHypernetworks}}>
            <TextualInversionsContext.Provider value={{textualInversions, setTextualInversions}}>
            <UpdateImagesContext.Provider value={{updateImages, setUpdateImages}}>
            <ImagesContext.Provider value={{images, setImages}}>
            <SocketContext.Provider value={{socket, setSocket}}>
            <MobileContext.Provider value={{mobile, setMobile}}>
            <EnableDragContext.Provider value={{enableDrag, setEnableDrag}}>
                <Context>
                    <Switch>
                        <Route exact path={["/", "/reverse-diffusion"]}><MainPage/></Route>
                        <Route path="*"><MainPage/></Route>
                    </Switch>
                </Context>
            </EnableDragContext.Provider>
            </MobileContext.Provider>
            </SocketContext.Provider>
            </ImagesContext.Provider>
            </UpdateImagesContext.Provider>
            </TextualInversionsContext.Provider>
            </HypernetworksContext.Provider>
            </LorasContext.Provider>
            </ReverseSortContext.Provider>
            </NSFWImagesContext.Provider>
            </ImageInputImagesContext.Provider>
            </NovelAIImagesContext.Provider>
            </NovelAINSFWImagesContext.Provider>
            </NovelAIImageInputImagesContext.Provider>
            </GeneratorContext.Provider>
            </HolaraAIImagesContext.Provider>
            </HolaraAINSFWImagesContext.Provider>
            </HolaraAIImageInputImagesContext.Provider>
            </SavedPromptsNovelAIContext.Provider>
            </SavedPromptsHolaraAIContext.Provider>
            </SavedPromptsContext.Provider>
            </ModelDirContext.Provider>
            </OutputDirContext.Provider>
        </div>
    )
}

export default App